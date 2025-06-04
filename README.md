# LESnets
Code for *" LESnets (Large-Eddy Simulation nets): Physics-informed neural operator for large-eddy simulation of turbulence"* [paper](https://www.sciencedirect.com/science/article/pii/S0021999125004085?via%3Dihub),[arxiv](https://arxiv.org/abs/2411.04502).


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
@article{ZHAO2025LESnets,
title = {LESnets (large-eddy simulation nets): Physics-informed neural operator for large-eddy simulation of turbulence},
journal = {Journal of Computational Physics},
volume = {537},
pages = {114125},
year = {2025},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2025.114125},
url = {https://www.sciencedirect.com/science/article/pii/S0021999125004085},
author = {Sunan Zhao and Zhijie Li and Boyu Fan and Yunpeng Wang and Huiyu Yang and Jianchun Wang},
keywords = {Fourier neural operator, Physics-informed neural operator, Turbulence, Large-eddy simulation},
abstract = {Acquisition of large datasets for three-dimensional (3D) partial differential equations (PDE) is usually very expensive. Physics-informed neural operator (PINO) eliminates the high costs associated with generation of training datasets, and shows great potential in a variety of partial differential equations. In this work, we employ physics-informed neural operator, encoding the large-eddy simulation (LES) equations directly into the neural operator for simulating three-dimensional incompressible turbulent flows. We develop the LESnets (Large-Eddy Simulation nets) by adding large-eddy simulation equations to two different data-driven models, including Fourier neural operator (FNO) and implicit Fourier neural operator (IFNO) without using label data. Notably, by leveraging only PDE constraints to learn the spatio-temporal dynamics, LESnets models retain the computational efficiency of data-driven approaches while obviating the necessity for data. Meanwhile, using LES equations as PDE constraints makes it possible to efficiently predict complex turbulence at coarse grids. We investigate the performance of the LESnets models with two standard three-dimensional turbulent flows: decaying homogeneous isotropic turbulence and temporally evolving turbulent mixing layer. In the numerical experiments, the LESnets models show similar accuracy as compared to traditional large-eddy simulation and data-driven models including FNO and IFNO, and exhibits a robust generalization ability to unseen regime of flow fields. By integrating a single set of flow data, the LESnets models can automatically learn the coefficient of the subgrid scale (SGS) model during the training of the neural operator. Moreover, the well-trained LESnets models are significantly faster than traditional LES, and exhibits comparable computational efficiency to the data-driven FNO and IFNO models. Thus, physics-informed neural operators have a strong potential for 3D nonlinear engineering applications.}
}
```
## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
