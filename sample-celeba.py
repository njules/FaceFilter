import os
import random

attribute = "young"

celeba_root = os.path.join(os.getcwd(), "celeba")

partitions = ["train", "val", "test"]

splits_file = os.path.join(celeba_root, "list_attr_celeba.txt")
split_by = "Young"
images_assignments = dict()
with open(splits_file) as f:
    header = None

    for i, line in enumerate(f):
        if i == 1:
            header = line.rstrip("\n").split()
        
        if i < 2:
            continue

        image, *attributes = line.split()
        attribute = attributes[header.index(split_by)]
        images_assignments[image] = int(attribute)

splits = ['A', 'B']
for partition in partitions:
    for split in ['A', 'B']:
        # os.rmdir(os.path.join(celeba_root, f"{partition}{split}"))
        os.mkdir(os.path.join(celeba_root, f"{partition}{split}"))

    src = os.path.join(celeba_root, partition)

    for image in os.listdir(src):
        split = 'B' if images_assignments[image] == -1 else 'A'
        dst = os.path.join(celeba_root, f"{partition}{split}")

        print(f"{os.path.join(src, image)} --> {os.path.join(dst, image)}")

        os.symlink(
            os.path.join(src, image),
            os.path.join(dst, image)
        )
