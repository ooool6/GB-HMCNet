import os
import glob
import numpy as np

h5_dir = r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\data\slices"
split_dir = r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\h5py-5-200"
out_dir = r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\lists"
os.makedirs(out_dir, exist_ok=True)

h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))
patient_to_files = {}
for f in h5_files:
    name = os.path.basename(f).split(".")[0]
    pid = int(name.split("_")[1])
    patient_to_files.setdefault(pid, []).append(name)

for fold_file in glob.glob(os.path.join(split_dir, "split_ind_fold*.npz")):
    fold_data = np.load(fold_file)
    fold_num = os.path.basename(fold_file).split("fold")[1].split(".")[0]

    for split_name in ['train', 'val', 'test']:
        list_path = os.path.join(out_dir, f"{split_name}_fold{fold_num}.list")
        with open(list_path, 'w') as f:
            for pid in fold_data[f"{split_name}_ind"]:
                for fname in sorted(patient_to_files[pid]):
                    f.write(fname + "\n")
        print(f"Saved {split_name} list: {list_path}")


