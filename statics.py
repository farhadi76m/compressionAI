import os
import re
import csv

input_dir = r'D:\SARBAZI\exp\CompressAI\results\statics'
output_csv = r'D:\SARBAZI\exp\CompressAI\results\statics_summary.csv'

def parse_filename(filename):
    # Example: bmshj2018_factorized_q1.txt or bmshj2018_factorized_relu_q1.txt
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Try to extract quality as the last 'q' + number part
    m = re.search(r'(.*)_((relu_)?q\d)$', name)
    if m:
        model_name = m.group(1)
        quality = m.group(2)
    else:
        # fallback: split by last underscore
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            model_name, quality = parts
        else:
            model_name, quality = name, ''
    return model_name, quality

def parse_averages(file_content):
    # look for line with: Average BPP: 0.455, Average PSNR: 30.16 dB
    avg_bpp = None
    avg_psnr = None
    for line in file_content.splitlines():
        if "Average BPP" in line and "Average PSNR" in line:
            m = re.search(r'Average BPP:\s*([\d.]+),\s*Average PSNR:\s*([\d.]+)', line)
            if m:
                avg_bpp = float(m.group(1))
                avg_psnr = float(m.group(2))
                break
    return avg_bpp, avg_psnr

rows = []
for fname in os.listdir(input_dir):
    if fname.endswith('.txt'):
        fullpath = os.path.join(input_dir, fname)
        with open(fullpath, 'r') as f:
            content = f.read()
        
        model_name, quality = parse_filename(fname)
        avg_bpp, avg_psnr = parse_averages(content)

        if avg_bpp is not None and avg_psnr is not None:
            rows.append([model_name, quality, avg_bpp, avg_psnr])
        else:
            print(f"Warning: averages not found in {fname}")

# Write CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['model_name', 'quality', 'avg_BPP', 'avg_PSNR'])
    writer.writerows(rows)

print(f"Summary CSV saved to {output_csv}")
