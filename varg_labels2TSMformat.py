import os
from pathlib import Path

label_files = sorted(Path("dataset/data_split/single_label/").glob("*.txt"))

path = "dataset/labels/"
if not os.path.exists(path):
    os.mkdir(path)

for txt in label_files:
    tmp = [x.strip().split(",") for x in open(txt)]
    labels = []
    for row in tmp[1:]:
        folder = os.path.join("dataset/videos/frames/", row[0])
        if os.path.exists(folder):
            num_frames = len(os.listdir(folder))
            labels.append(" ".join([row[0], str(num_frames), row[1]]))
    print(f"Num labels before: {len(tmp)-1}, Num labels after: {len(labels)}")
    with open(os.path.join(path, txt.name), "w") as f:
        f.write("\n".join(labels))
