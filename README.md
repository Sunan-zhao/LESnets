# LESnets
Code for *"Latent Neural PDE Solver: a reduced-order modelling framework for partial differential equations"*  [arxiv](https://arxiv.org/abs/2411.04502).


<div style style=”line-height: 20%” align="center">
<h3> Models prediction on 3D decaying homogeneous isotropic turbulence </h3>
<img src="https://github.com/Sunan-zhao/LESnets/blob/master/DHIT_sample5_3tau.gif" width="600">


  
## Abstrct

Acquisition of large datasets for three-dimensional (3D) partial differential equations are usually very expensive. Physics-informed neural operator (PINO) eliminates the high costs associated with generation of training datasets, and shows great potential in a variety of partial differential equations. In this work, we employ physics-informed neural operator, encoding the large-eddy simulation (LES) equations directly into the neural operator for simulating three-dimensional incompressible turbulent flows. We develop the LESnets (Large-Eddy Simulation nets) by adding large-eddy simulation equations to two different data-driven models, including Fourier neural operator (FNO) and implicit Fourier neural operator (IFNO) without using label data. Notably, by leveraging only PDE constraints to learn the spatio-temporal dynamics problem, LESnets retains the computational efficiency of data-driven approaches while obviating the necessity for data. Meanwhile, using large-eddy simulation equations as PDE constraints makes it possible to efficiently predict complex turbulence at coarse grids. We investigate the performance of the LESnets with two standard three-dimensional turbulent flows: decaying homogeneous isotropic turbulence and temporally evolving turbulent mixing layer. In the numerical experiments, the LESnets model shows a similar or even better accuracy as compared to traditional large-eddy simulation and data-driven models of FNO and IFNO. By integrating a small amount of LES data, the LESnets model is able to learn the hyperparameters of the subgrid model during the training of the neural operator. Moreover, the well-trained LESnets is significantly faster than traditional LES, and has a similar efficiency as the data-driven FNO and IFNO models. Thus, physics-informed neural operators have a strong potential for 3D nonlinear engineering applications.

## Datasets

We plan to make the code and dataset public once the manuscript is accepted. 


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
