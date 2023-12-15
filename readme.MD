# Singular Conotrl Solver
## Singular Control of (Reflected) Brownian Motion: A Computational Method Suitable for Queueing Applications
requirement: Numpy and Tensorflow 2

Usage:
```
python3 main.py --config_path=configs/tandem.json 
```
*tandem.json* can be replaced to other json file.



After running once, the data will be automatically dumped. Then, one can use the following commend
```
python3 main.py --config_path=configs/tandem.json --dump=False
```

### Configs
*tandem.json* is for the tandem queues example;
*singularcontrol.json* is for the criss-cross example;
*bigstep.json* is for the three-station example;
*tandemHD.json* is for many queues in series;

The code is adapted from https://github.com/MoZhou1995/DeepPDE_ActorCritic.

### MDP
We implement the exact MDP solutions for 2D tandem queue example and criss-cross network.

### Method from Kushner and Martins
we implement the method from [1].
[1] Kushner, Harold J., and Luiz Felipe Martins. "Numerical methods for stochastic singular control problems." SIAM journal on control and optimization 29.6 (1991): 1443-1475.