"""
This module has a collection of utility funcions that may be used across multiple other modules.
"""

import ast
import math


# Receive a number as a string. Returns that number as an int or as a float, depending on the case.
# For instance, num(2) = 2 and num(1.0) = 1.0
def num(s):
    return ast.literal_eval(s)


# Computes the density function of the normal distribution at x,
# given the mean value and the variance (square of standard deviation).
def gaussian(mean, variance, x):
    return math.exp(-(x-mean)**2/(2.0*variance))/math.sqrt(2*math.pi*variance)


# Returns true iff the attribute is categorical.
# 'attribute' is the number of the attribute.
def is_categorical(attribute):
    return attribute == 10 or attribute == 11


# Returns the list of possible values of a given categorical attribute.
def categorical_attribute_values(attribute):
    if is_categorical(attribute):
        if attribute == 10:
            return list(range(4))
        elif attribute == 11:
            return list(range(40))
        else:
            raise Exception('Utils.categorical_attribute_values: categorical attribute should be attribute 10 and 11')
    else:
        raise Exception('Utils.categorical_attribute_values: attribute must be categorical')


# Returns the number of possible values of a given categorical attribute.
def categorical_attribute_values_number(attribute):
    if is_categorical(attribute):
        if attribute == 10:
            return 4
        elif attribute == 11:
            return 40
        else:
            raise Exception('Utils.categorical_attribute_values_number: categorical attribute should be' +
                            'attribute 10 and 11')
    else:
        raise Exception('Utils.categorical_attribute_values_number: attribute must be categorical')

