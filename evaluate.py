import argparse
import os
from glob import glob
import numpy as np
from PIL import Image
import onnxruntime as ort
import pickle, zlib
from skimage.metrics import peak_signal_noise_ratio as psnr

QUANT_BITS = 6

def quantize(x, bits):
    M = np.max(np.abs(x))
    scale = (2**(bits - 1) - 1) / M if M != 0 else 1.0
    q = np.round(x * scale).astype(np.int8)
    u = (q + 2**(bits - 1)).astype(np.uint8)
    return u, scale

def dequantize(u, scale, bits):
    q = u.astype(np.int16) - 2**(bits - 1)
    return q.astype(np.float32) / scale

def entropy_encode_symbol(symbol, cdf, cdflen, cdfoff):
    idx = max(0, min(symbol, len(cdf) - 1))
    cum_freq_low = 0 if idx == 0 else cdf[idx - 1]
    cum_freq_high = cdf[idx]
    return cum_freq_low, cum_freq_high, cdflen

def entropy_decode_symbol(cum_freq, cdf, cdflen, cdfoff):
    left, right = 0, len(cdf) - 1
    while left < right:
        mid = (left + right) // 2
        if cdf[mid] <= cum_freq:
            left = mid + 1
        else:
            right = mid
    return left

def entropy_encode_tensor(tensor, cdf, cdflen, cdfoff):
    flat = tensor.flatten()
    encoded_data = []
    for symbol in flat:
        adjusted_symbol = int(symbol) - cdfoff if cdfoff is not None else int(symbol)
        adjusted_symbol = max(0, min(adjusted_symbol, len(cdf) - 1))
        low, high, total = entropy_encode_symbol(adjusted_symbol, cdf, cdflen, cdfoff)
        encoded_data.append((low, high, total))
    return zlib.compress(pickle.dumps(encoded_data), level=9)

def entropy_decode_tensor(compressed_data, shape, cdf, cdflen, cdfoff):
    decompressed = zlib.decompress(compressed_data)
    encoded_data = pickle.loads(decompressed)
    decoded_symbols = []
    for low, high, total in encoded_data:
        cum_freq = (low + high) // 2
        symbol = entropy_decode_symbol(cum_freq, cdf, cdflen, cdfoff)
        if cdfoff is not None:
            symbol += cdfoff
        decoded_symbols.append(symbol)
    return np.array(decoded_symbols, dtype=np.uint8).reshape(shape)

def process_image(image_path, encoder, decoder, cdf, cdflen, cdfoff, output_dir):
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, np.float32) / 255.0
    arr = img_arr.transpose(2, 0, 1)[None]
    orig_shape = arr.shape

    enc_out = encoder.run(None, {encoder.get_inputs()[0].name: arr})
    parts, total_original_size, total_compressed_size = [], 0, 0

    for feat in enc_out:
        u, scale = quantize(feat, QUANT_BITS)
        original_size = u.nbytes
        total_original_size += original_size
        comp = zlib.compress(u.tobytes(), 9)
        total_compressed_size += len(comp)
        parts.append({'shape': u.shape, 'scale': scale, 'data': comp})


    packed = {'orig_shape': orig_shape, 'parts': parts}
    comp_path = os.path.join(output_dir, os.path.basename(image_path).replace('.png', '_compressed.bin'))
    with open(comp_path, 'wb') as f:
        pickle.dump(packed, f)

    # Decode
    recs = []
    for p in parts:
        u = np.frombuffer(zlib.decompress(p['data']), dtype=np.uint8).reshape(p['shape'])
        recs.append(dequantize(u, p['scale'], QUANT_BITS))


    inputs = {inp.name: recs[i] for i, inp in enumerate(decoder.get_inputs())}
    out = decoder.run(None, inputs)[0][0].transpose(1, 2, 0)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)

    # Save image
    rec_img_path = os.path.join(output_dir, os.path.basename(image_path).replace('.png', '_recon.png'))
    Image.fromarray(out).save(rec_img_path)

    bpp = (total_compressed_size * 8) / (img.width * img.height)
    image_psnr = psnr((img_arr * 255).astype(np.uint8), out, data_range=255)

    return {
        'image': os.path.basename(image_path),
        'bpp': round(bpp, 3),
        'psnr': round(image_psnr, 2),
        'original_size': total_original_size,
        'compressed_size': total_compressed_size
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory with .png images")
    parser.add_argument("output_dir", help="Directory to save results")
    parser.add_argument("model_path", help="Path to model checkpoint")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = sorted(glob(os.path.join(args.input_dir, "*.png")))

    encoder = ort.InferenceSession(os.path.join(args.model_path,"encoder.onnx"))
    decoder = ort.InferenceSession(os.path.join(args.model_path,"decoder.onnx"))

    try:
        with open(os.path.join(args.model_path,"info.pkl"), "rb") as f:
            model_info = pickle.load(f)
        cdf, cdflen, cdfoff = model_info['cdf'], model_info['cdflen'], model_info['cdfoff']
        print("Loaded CDF info.")
    except FileNotFoundError:
        print("CDF file not found. Using zlib fallback.")
        cdf, cdflen, cdfoff = None, None, None

    results = []
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        res = process_image(img_path, encoder, decoder, cdf, cdflen, cdfoff, args.output_dir)
        results.append(res)

    print("\n=== Compression Results ===")
    for r in results:
        print(f"{r['image']}: BPP={r['bpp']}, PSNR={r['psnr']} dB")

    avg_bpp = np.mean([r['bpp'] for r in results])
    avg_psnr = np.mean([r['psnr'] for r in results])
    print(f"\nAverage BPP: {avg_bpp:.3f}, Average PSNR: {avg_psnr:.2f} dB")

    with open(os.path.join(args.output_dir, "results_summary.txt"), "w") as f:
        for r in results:
            f.write(f"{r['image']}: BPP={r['bpp']}, PSNR={r['psnr']} dB\n")
        f.write(f"\nAverage BPP: {avg_bpp:.3f}, Average PSNR: {avg_psnr:.2f} dB\n")

if __name__ == "__main__":
    main()
