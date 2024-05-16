'''
Date: May 6th 2022
Authors: 
Njaka ANDRIAMANDRATOMANANA, Elie CHEDEMAIL,
Adéchola KOUANDE, Rémi LELUC, Quyen NGUYEN
This file contains some tool functions to work on 
Rq = Zq[X]/(X^N + 1)
'''
# List of functions
# 1.draw_from_binary_vec
# 2.draw_from_integer_vec
# 3.draw_from_normal_vec
# 4.draw_from_binary
# 5.draw_from_integer
# 6.draw_from_normal
# 7.mod
# 8.base_decomp

# import libraries
import numpy as np
from math import log,floor


###########################################
# Tool functions to draw vectors (for LWE)#
###########################################
def draw_from_binary_vec(n):
    """ function to draw binary secret keys
    Params:
    @n  (int): security parameter (lattice dimension)
    Returns:
    """
    return np.random.randint(0, 2, n).astype(int)


def draw_from_integer_vec(n, q):
    """ function to draw vector in Z/qZ
    Params:
    @n (int): security parameter (lattice dimension)
    @q (int): size of quotient space Z/Zq
    Returns:
    uniform coefficients in Z/qZ
    """
    return np.random.randint(0, q, n).astype(int) % q


def draw_from_normal_vec(n, q, loc=0, scale=2):
    """ function to draw errors e from discrete Gaussian
    Params:
    @n       (int): security parameter (lattice dimension)
    @q       (int): size of quotient space Z/Zq
    @loc   (float): mean of gaussian (defaut 0 for centered)
    @scale (float): std of gaussian 
    Returns:
    gaussian coefficients in Z/qZ
    """
    return np.random.normal(loc, scale, n).astype(int) % q

################################################
# Tool functions to draw polynomials (for RLWE)#
################################################
def draw_from_binary(n):
    """ function to draw binary secret keys
    Params:
    @n  (int): security parameter (lattice dimension)
    Returns:
    poly1d object of degree n with binary coefficients
    """
    return np.poly1d((np.random.randint(0, 2, n+1).astype(int)))


def draw_from_integer(n, q):
    """ function to draw polynomial in Z/qZ
    Params:
    @n (int): security parameter (lattice dimension)
    @q (int): size of quotient space Z/Zq
    Returns:
    poly1d object of degree n with uniform coefficients in Z/qZ
    """
    return np.poly1d(np.random.randint(0, q, n+1).astype(int) % q)


def draw_from_normal(n, q, loc=0, scale=2):
    """ function to draw errors e from discrete Gaussian
    Params:
    @n       (int): security parameter (lattice dimension)
    @q       (int): size of quotient space Z/Zq
    @loc   (float): mean of gaussian (defaut 0 for centered)
    @scale (float): std of gaussian 
    Returns:
    poly1d object of degree n with gaussian coefficients in Z/qZ
    """
    return np.poly1d(np.round(np.random.normal(loc, scale, n+1)).astype(int) % q)

#######################################
# Function to get elements back in Rq#
######################################
def mod(poly, q, poly_modulus):
    """ divide polynomial by poly_modulus and take coefficients modulo q
    Params:
    @poly         (poly1d): input polynomial
    @q               (int): size of quotient space Z/Zq
    @poly_modulus (poly1d): quotient polynomial
    Returns:
    poly1d object of degree n with coefficients in Z/qZ
    """
    return np.poly1d(np.floor(np.polydiv(poly, poly_modulus)[1]).astype(int) % q)

# Function to perform decomposition in base B for polynomial
def base_decomp(poly, q, B):
    """ Perform B-decomposition of poly in R_q
    k = log_B(q), poly = sum_{i=0}^{k-1} p_i B^i, p_i in R_q
    Params:
    @poly (poly1d): input polynomial
    @q       (int): size of quotient space Z/Zq
    @B       (int): base for decomposition
    Returns:
    @results (array of np.poly1d): list of all the p_i
    """
    k = floor(log(q,B))
    result = np.zeros(shape=k,dtype=object)
    for i in range(k):
        result[i] = np.poly1d(np.floor(poly / B ** i).astype(int) % B)
    return result

# Function to perform decomposition in base B
def base_decomp_vec(vec, q, Br):
    """ Perform B-decomposition of vector in Z_q
    k = log_B(q), a = sum_{i=0}^{k-1} a_i Br^i, a_i in Z_q
    Params:
    @vec     (int): input vector
    @q       (int): size of quotient space Z/Zq
    @Br      (int): base for decomposition
    Returns:
    @results (array of np.poly1d): list of all the p_i
    """
    l = floor(log(q,Br))
    result = np.zeros(shape=l)
    for i in range(l):
        result[i] = np.floor(vec / Br ** i).astype(int) % Br
    return result.astype(int)

