#!/bin/sh
python Main.py --s 'CVXOPT' --C 'gaussian' --B 'identity' --d 'crossfeeding'
python Main.py --s 'CVXOPT' --C 'gaussian' --B 'null' --d 'crossfeeding'
python Main.py --s 'CVXOPT' --C 'gaussian' --B 'block' --d 'crossfeeding'
python Main.py --s 'CVXOPT' --C 'gaussian' --B 'circulant' --d 'crossfeeding'