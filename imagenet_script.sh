#$ -l mem=4G
#$ -l h_rt=01:30:00
#$ -l gpu=2
#$ -pe smp 8
#$ -N imagenet_ViTstar_12layers
#$ -R y
#$ -ac allow=E,F
#$ -S /bin/bash
#$ -wd /home/ucabgj1/Scratch/smoothtransformer/how-do-vits-work
#$ -j y

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate smoothtransformer

date
nvidia-smi
python classification_imagenet.py