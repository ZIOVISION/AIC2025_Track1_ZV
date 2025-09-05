import os
import glob

track_txt_folder = '/perception/dataset/PhysicalAI-SmartSpaces/MTMC_Tracking_2025_outputs/tracking/overlap01_val_best_track_out'
output_path = 'best_overlap01_track1.txt'


txt_files = glob.glob(os.path.join(track_txt_folder, '*.txt'))
print(len(txt_files), "txt files found in", track_txt_folder)
merged_lines = []
for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        merged_lines.extend(lines)

with open(output_path, 'w') as f:
    for line in merged_lines:
        f.write(line)