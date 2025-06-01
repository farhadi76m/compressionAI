import argparse
import os
import torch
import torchvision.transforms as T
import pickle
from PIL import Image
import onnxruntime
import numpy as np
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_factorized_relu,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    cheng2020_attn,
    mbt2018,
    mbt2018_mean
)

MODEL_TYPES = {
    'bmshj2018-factorized': bmshj2018_factorized,
    'bmshj2018-factorized-relu': bmshj2018_factorized_relu,
    'bmshj2018-hyperprior': bmshj2018_hyperprior,
    'cheng2020-anchor': cheng2020_anchor,
    'cheng2020-attn': cheng2020_attn,
    'mbt2018': mbt2018,
    'mbt2018-mean': mbt2018_mean
}

class ModelConverter:
    def __init__(self, model_type, quality=1):
        self.device = 'cpu'  # Converting to ONNX should be done on CPU
        
        # Initialize the model
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type. Choose from: {', '.join(MODEL_TYPES.keys())}")
        
        self.model = MODEL_TYPES[model_type](quality=quality, pretrained=True).eval().to(self.device)

    def _convert_encoder(self, input_image, output_path, model_info_path):
        """Convert encoder (g_a) to ONNX format"""
        # Prepare input image
        transform = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor()
        ])
        x = Image.open(input_image)
        x = transform(x).unsqueeze(0)
        torch.onnx.export(
            self.model.g_a,
            x,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['encoded'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'encoded': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        print(f"Encoder model saved to {output_path}")
        entropy_bottleneck = self.model.entropy_bottleneck 
         
        cdf = entropy_bottleneck._quantized_cdf.clone().cpu().numpy()
        cdflen = entropy_bottleneck._cdf_length.clone().cpu().numpy()
        cdfoff = entropy_bottleneck._offset.clone().cpu().numpy()   
        quant_layer = entropy_bottleneck.__class__.__name__
        
        
        
        dict_cdf = {}
        dict_cdf["cdf"] = cdf
        dict_cdf["cdflen"] = cdflen
        dict_cdf["cdfoff"] = cdfoff

        pickle.dump(dict_cdf, open(model_info_path, "wb"))
        print("[INFO] wrote decoder info: ", model_info_path)
        
    def _convert_decoder(self, output_path, input_shape=(1, 192, 32, 32)):
        print("""Convert decoder (g_s) to ONNX format""")
        print(f"decoder input shape is {input_shape}")
        
        x = torch.randn(input_shape)

        # class DecoderWrapper(torch.nn.Module):
        #     def __init__(self, decoder):
        #         super().__init__()
        #         self.decoder = decoder

        #     def forward(self, x):
        #         return self.decoder(x)

        # wrapper = DecoderWrapper(self.model.g_s)
        # wrapper.eval()

        torch.onnx.export(
            self.model.g_s,
            x,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['encoded'],
            output_names=['decoded'],
            dynamic_axes={
                'encoded': {0: 'batch_size', 2: 'height', 3: 'width'},
                'decoded': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        print(f"Decoder model saved to {output_path}")

    def convert(self, input_image, export_dir):
        """Convert both encoder and decoder to ONNX format"""
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate output paths
        encoder_path = os.path.join(export_dir, "encoder.onnx")
        decoder_path = os.path.join(export_dir, "decoder.onnx")
        model_info_path = os.path.join(export_dir, "info.pkl")
        
        print("Converting encoder model...")
        self._convert_encoder(input_image, encoder_path, model_info_path)
        
        print("\nConverting decoder model...")
        
        encoder = onnxruntime.InferenceSession(encoder_path)
        img = Image.open(input_image)
        img = img.resize([512,512])
        img = np.array(img).transpose([2,0,1])[None,...].astype(np.float32)
        img = img/255.0
        
        encoder_input = {encoder.get_inputs()[0].name: img}
        encoded = encoder.run(None, encoder_input)[0]
        self._convert_decoder(decoder_path, encoded.shape)
        
        print(f"\nAll models exported to: {export_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert CompressAI model to ONNX format")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to input image")
    parser.add_argument("--export-dir", type=str, required=True, 
                        help="Directory to save the ONNX models")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=list(MODEL_TYPES.keys()),
                        help="Type of compression model to use")
    parser.add_argument("--quality", type=int, default=1,
                        help="Quality factor (1-8)")
    
    args = parser.parse_args()

    try:
        converter = ModelConverter(
            model_type=args.model_type,
            quality=args.quality
        )
        converter.convert(args.input, args.export_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()