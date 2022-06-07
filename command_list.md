# For CycleGan

# Create the environment
conda env create -n mla39 -f FaceAging-by-cycleGAN/environment.yml
source activate mla39
pip install torchvision
pip install -r FaceAging-by-cycleGAN/requirements.txt
conda install cython

# Think for running the code
git clone https://github.com/njules/FaceFilter
cd FaceFilter
git clone https://github.com/jiechen2358/FaceAging-by-cycleGAN.git
module load conda
module load intel/python/3/2019.4.088

rm FaceAging-by-cycleGAN/environment.yml
mv environment_cyclegan.yml FaceAging-by-cycleGAN/environment.yml

# Upload the dataset and after 
[Link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) to download the dataset.
# Load the dataset from a local terminal with the command:
scp -r /Users/mattiadutto/Downloads/dataset_path  mla_group_13@legionlogin.polito.it:/home/mla_group_13/FaceAging-by-cycleGAN/
# Unzip e process it.
unzip -qq img_align_celeba.zip

python split-celeba.py
python sample-celeba.py

# Move pretrained model
mv trained_model/9_wiki_fine_tune_male/* checkpoints/9_wiki_fine_tune_male/
# After move the checkpoint we have to rename it
mv latest_net_D_A.pth 200_net_D_A.pth
mv latest_net_D_B.pth 200_net_D_B.pth
mv latest_net_G_A.pth 200_net_G_A.pth
mv latest_net_G_B.pth 200_net_G_B.pth

# Move the data_loader_cyclegan, if we are on the FaceFilter directory
mv data_loader_cyclegan.py ./FaceAging-by-cycleGAN/data/__init__.py
# Move the train_cyclegan, if we are on the FaceFilter directory
mv train_cyclegan.py ./FaceAging-by-cycleGAN/train.py

# For the first time, create the output folder
mkdir experiments_output