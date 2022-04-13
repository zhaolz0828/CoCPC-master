#!/usr/bin/env python
"""
@Time ： 2022/3/24 15:04
@Auth ： 赵凌子
@File ： sigmod_test.py
@IDE ：  PyCharm
@Motto： After all,tomorrow is another day.
"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


sigmoid_inputs = np.arange(-10, 10, 0.1)
sigmoid_outputs = sigmoid(sigmoid_inputs)
print("Sigmoid Function Input :: {}".format(sigmoid_inputs))
print("Sigmoid Function Output :: {}".format(sigmoid_outputs))

plt.plot(sigmoid_inputs, sigmoid_outputs)
plt.xlabel("Sigmoid Inputs")
plt.ylabel("Sigmoid Outputs")
plt.show()