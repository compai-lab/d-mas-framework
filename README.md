# MAD-DL

Deep Learning Framework for the Deep Medical Anomaly Segmentation Seminar

[![Open Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compai-lab/d-mas-framework/blob/main/demo.ipynb)

![Framework overview](./iml_dl.png)


# Installation guide: 

0). Set up wandb. (https://docs.wandb.ai/quickstart)
 *  Sign up for a free account at https://wandb.ai/site and then login to your wandb account.
 * Login to wandb with `wandb login`
 * Paste the API key when prompted. You will find your API key here: https://wandb.ai/authorize. 
 
1). Clone d-mas framework to desired location 
 * `git clone https://github.com/compai-lab/d-mas-framework.git *TARGET_DIR*`

2). Create a virtual environment with the needed packages (use conda_environment-osx.yaml for macOS)
```
cd ${TARGET_DIR}/d-mas-framework
conda env create -f conda_environment.yaml
source activate mas_py308 *or* conda activate mas_py308
```

3). Install pytorch
* *with cuda*: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* *w/o cuda*:
```
pip3 install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

4). Run the demo script: 
```
python core/Main.py --config_path projects/Autoencoders/d-ae.yaml
```


5). _Optional_: Clone the projects folder to your own Github:

```
cd ${TARGET_DIR}/iml-dl/projects
git init
git remote add origin $URL_TO_YOUR_REPO
git branch -M main
git commit -m "first commit"
git push -u origin main
```

# That's it, enjoy! :rocket:
