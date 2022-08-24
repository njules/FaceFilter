
# Face Aging with Generative Adversarial Network Project Report

This project was developed for 01URXOV - Machine Learning in Applications (spring semester 2022) of Politecnico di Torino.

Authors: Mattia Dutto, Julian Neubert, Simone Alberto Peirone.

## Contributions
The starting point is the repo [**FaceAgin-by-cycleGAN**](https://github.com/jiechen2358/FaceAging-by-cycleGAN.git) that it used for the baseline approach. 
The starting point of AE-StyleGAN is taken from [**repo**](https://github.com/phymhan/stylegan2-pytorch) 

Our contribution mainly focuses on the use of CelebA dataset on the two approaches and on the Cycle-StyleGAN, that is based on the previously cited AE-StyleGAN. 

## Scripts details:
* image_folder_cyclegan.py: it allows to use CelebA dataset to the FaceAging-by-cycleGAN architecture. There are basically custom files of the one present in the repo.
* split-celaba.py: it allows to create the test, validation and train split of the selected dataset.
* sample-celeba.py: it allows use to split the dataset partitions (*train*, *validation*, *test*) based on a attribute by creating 2 sub-folders (in this case the **young** and **old** one). This script does not copy the source images to the destination file, it will just create symlinks.
* train_cyclegan.py: is the train.py customized for our task. 
* envinroments_cyclegan.yml: is for install all the dependencies of Face-Aging-by-cycleGAN. 

## Scripts folder
Inside this folder you can find the examples of the sbatch scripts for the different networks on the [cluster of Politecnico of Turin](https://hpc.polito.it).
* run_cycle_gan.sbatch: Cycle-GAN training
* run_ae_stylegan.sbatch: AE-StyleGAN training
* run_ae_stylegan_ablation.sbatch: hyper-parameter search on AE-StyleGAN
* run_ae_cycle_stylegan.sbatch: Cycle-StyleGAN training.

## List of task to do before running any experiment.
* Clone this repo and clone inside the [FaceAging-by-cycleGAN repo](https://github.com/jiechen2358/FaceAging-by-cycleGAN.git)
* Create the environment: with torchvision and the requirement present on FaceAging-by-cycleGAN/requirements.txt
    ```bash
    conda env create -n fa -f FaceAging-by-cycleGAN/environment.yml
    source activate fa
    pip install torchvision
    pip install -r FaceAging-by-cycleGAN/requirements.txt
    conda install cython
    ```
* CelebA dataset:
  * Download the dataset from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg).
  * Unzip the dataset
  * Run the split-celeba.py script
  * Run the sample-celeba.py script
  ```bash
  unzip -qq img_align_celeba.zip

  python split-celeba.py
  python sample-celeba.py
  ```
* Move pre-trained weights to the correct place and rename them (the 9_wiki_fine_tune_male pre-trained model was used and the latest part must be changed with 200)
    ```bash
    # note: You need to be on the home directory of FaceAging-by-cycleGAN repo.
    
    mv trained_model/9_wiki_fine_tune_male/* checkpoints/9_wiki_fine_tune_male/

    mv latest_net_D_A.pth 200_net_D_A.pth
    mv latest_net_D_B.pth 200_net_D_B.pth
    mv latest_net_G_A.pth 200_net_G_A.pth
    mv latest_net_G_B.pth 200_net_G_B.pth
    ```
* Move the custom files to the correct directory.
  ```bash
  # note: position yourself on the home directory of this repo.
  mv data_loader_cyclegan.py ./FaceAging-by-cycleGAN/data/__init__.py
  ```

## Example of test command for Cycle-GAN 
This command allows to generate one of the results that you can see on the repo.
```bash 
python test.py --dataroot ../celeba \
               --name 9_wiki_fine_tune_male/05_epochs \
               --model cycle_gan  \
               --epoch end \
               --gpu_ids -1 \
               --num_test 8
```
All the options are describe on *FaceAging-by-cycleGAN/options*.