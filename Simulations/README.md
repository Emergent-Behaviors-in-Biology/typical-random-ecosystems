# Numerical Simulations in Typical Random Ecosystems
## Getting Started

This implementation requires [Eco_functions](https://github.com/Wenping-Cui/Eco_functions). Download it and install with
``` bash
$ pip install e .
```

## Running without Crossfeeding

```bash
$ python Main_simulation.py --B 'identity' --C 'gaussian'  --d 'quadratic' 

```

## Running with Crossfeeding

```bash
$ python Simulations_Crossfeeding.py --B  'identity' --C 'binomial'   --d 'quadratic' 
```
### Arguments

| Argument &nbsp; &nbsp; &nbsp; &nbsp; | Description | Values |
| :---         |     :---      |          :--- |
| --B        |     Type of Engineered matrix      |  'gaussian', 'uniform'，‘binomial’ |
| --C     | Type of Noise       | 'identity', 'null', 'circulant' and 'block'     |
| --d   | Type of Resource dynamics     | 'quadratic', 'linear' |
| epsilon  | Amplitue of noise    | b, pc or sigc |