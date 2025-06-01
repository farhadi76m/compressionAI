# 📦 CompressAI ONNX Converter & Evaluator

This project provides tools for converting deep image compression models built with [CompressAI](https://github.com/InterDigitalInc/CompressAI) from PyTorch to ONNX, and evaluating their performance on the Kodak dataset.

---

## 🚀 Features

- 🔄 Convert trained PyTorch compression models to ONNX format
- 🧪 Evaluate ONNX models using the Kodak dataset
- 📊 Compute image quality metrics (PSNR) and bitrate (bpp)

---

## 📁 Datast : 
├── convert_onnx.py # Script to convert PyTorch model to ONNX
├── evaluate.py # Script to evaluate ONNX model on Kodak dataset
├── models/ # Directory for storing ONNX models
├── data/ # Kodak dataset images (should be placed here)
└── README.md

## 🛠 Requirements

- Python 3.8+
- PyTorch
- CompressAI
- ONNX
- ONNX Runtime
- NumPy
- OpenCV or Pillow
  
Install dependencies:
```
pip install compressioai
pip install onnx
pip install onnxruntime
```

🔧 Usage
✅ Convert PyTorch model to ONNX :
```
python .\development\convert_onnx.py \
  --input INPUT_IMAGE \
  --export-dir EXPORT_DIR \
  --model-type MODEL_TYPE \
  --quality Q \

```


`--input`: Path to dataset 
`--export-dir`: Destination ONNX model path
`--model-type` : Model types in compressionAI
`--quality` : quality of models (1 to 6)

📊 Evaluate ONNX model on Kodak dataset : 
```
python evaluate.py \
  input_dir \ # List of images 
  output_dir \ # For saving results
  model_pth # model directory (encoder.onnx, decoder.onxx ,[info.pkl])

```

📈 Evaluation Metrics

The evaluation computes:

    PSNR – Peak Signal-to-Noise Ratio

    SSIM – Structural Similarity Index

    bpp – Bits per pixel (bitrate)


🤝 Acknowledgments

    [CompressAI](https://github.com/InterDigitalInc/CompressAI)-based PyTorch models
    [ONNX](https://onnx.ai/)
    [Kodak Image Dataset](http://r0k.us/graphics/kodak/)

    Kodak image dataset
