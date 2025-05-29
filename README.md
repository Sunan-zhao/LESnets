# LESnets
Code for *" LESnets (Large-Eddy Simulation nets): Physics-informed neural operator for large-eddy simulation of turbulence"*  [arxiv](https://arxiv.org/abs/2411.04502).


<h3> LESnets prediction on 3D decaying homogeneous isotropic turbulence </h3>
<img src="https://github.com/Sunan-zhao/LESnets/blob/main/assets/DHIT_sample5_3tau.gif" width="600">


<img src="https://github.com/Sunan-zhao/LESnets/blob/main/assets/LESnets_model.jpg" width="720">


## Requirements
- Python 3.9.13
- Pytorch 1.12.1+cu116
- numpy

## Datasets
The datasets for training, testing and predicting can be downloaded at: https://www.kaggle.com/datasets/sunanzhao123/lesnets-datasets.

## Experiments
Please refer to the yaml file in each experiment for detailed hyperparameter settings.

### 1. Decaying homogeneous isotropic turbulence (DHIT)
To run LESnets models or neural operator models for DHIT, use, e.g.,
```bash 
python train_pino.py --config config/DHIT/PI_FNO_L6_W80_M12_data5000_g0_gp0.yaml
```
To run neural operator models for TML, use, e.g.,

```bash 
python train_pino.py --config config/DHIT/FNO_L6_W80_M12_data5000_g1_gp0.yaml
```

### 2. Turbulent mixing layer (TML)
To run LESnets models or neural operator models for TML, use, e.g.,
```bash 
python train_pino.py --config config/TML/PI_FNO_L20_W150_M12_data2000_g0_gp0.yaml
```
To run neural operator models for TML, use, e.g.,

```bash 
python train_pino.py --config config/TML/IFNO_L20_W150_M12_data2000_g1_gp0.yaml
```
### 3. Automatically learn SGS coefficient and some analysis
To run LESnets with automatically learn SGS coefficient, use the .yaml file with "cslr". e.g., 
```bash 
python train_pino.py --config config/DHIT/PI_FNO_L6_W80_M12_data5000_g0_gp05_fDNS_cslr1e5.yaml
```
To run LESnets with different model parameter, use the .yaml file with different "L", "W", "M";

To run LESnets with different number of initial fields, use the .yaml file with different "data";

To run LESnets with with both data and PDE loss, use the .yaml file with different "g";


## Citation

If you use our models, data or code for academic research, you are encouraged to cite the following paper:

```
@article{zhao2024lesnets,
  title={LESnets (Large-Eddy Simulation nets): Physics-informed neural operator for large-eddy simulation of turbulence},
  author={Zhao, Sunan and Li, Zhijie and Fan, Boyu and Wang, Yunpeng and Yang, Huiyu and Wang, Jianchun},
  journal={arXiv preprint arXiv:2411.04502},
  year={2024}
}
```
## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
