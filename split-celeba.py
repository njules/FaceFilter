import os

celeba_root = "./celeba/"
img_root = "img_align_celeba"
partition_file = "./celeba/list_eval_partition.txt"

partitions = ["train", "val", "test"]

for partition in partitions:
    partition = os.path.join(celeba_root, partition)
    if os.path.exists(partition):
        os.rmdir(partition)
    os.mkdir(partition)

with open(partition_file) as f:
    for idx, line in enumerate(f):
        if idx == 0: continue

        image, partition_idx = line.rstrip("\n").split(" ")
        partition = partitions[int(partition_idx)]

        os.symlink(
            os.path.join(os.getcwd(), celeba_root, img_root, image),
            os.path.join(celeba_root, partition, image)
        )
        # print(f"{image} -> {partition}")

