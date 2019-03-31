# Numerical Simulations in Typical Random Ecosystems
## Getting Started

This implementation requires [Eco_functions](https://github.com/Wenping-Cui/Eco_functions). For a new version, you can download it and install with
``` bash
$ pip install e .
```

## Running with Convex Optimization Solver

```bash
$ python main.py --B 'identity' --C 'gaussian'  --d 'quadratic'  --s 'CVXOPT'

```

## Running with ODE solver

```bash
$ python main.py --B 'identity' --C 'gaussian'  --d 'quadratic'  --s 'ODE'
```
### Arguments

| Argument &nbsp; &nbsp; &nbsp; &nbsp; | Description | Values |
| :---         |     :---      |          :--- |
| --B        |     Type of Engineered matrix      |  'gaussian', 'uniform'，‘binomial’ |
| --C     | Type of Noise       | 'identity', 'null', 'circulant' and 'block'     |
| --d   | Type of Resource dynamics     | 'quadratic', 'linear' ,'crossfeeding'|
| --s   | Type of Solver     | 'CVXOPT', 'ODE'|
| epsilon  | Amplitue of noise    | sigc, b, pc |