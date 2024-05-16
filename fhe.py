'''
Date: May 6th 2022
Authors: 
Njaka ANDRIAMANDRATOMANANA, Elie CHEDEMAIL,
Adéchola KOUANDE, Rémi LELUC, Quyen NGUYEN
This file contains some functions to perform FHE
'''
# List of functions
# 1.LWE
# 2.inv_LWE
# 3.RLWE
# 4.inv_RLWE
# 5.RLWE_prod
# 6.prod_ext

# import libraries
import numpy as np
from math import log,floor

from utils import draw_from_integer_vec, draw_from_normal_vec
from utils import draw_from_integer, draw_from_normal
from utils import mod, base_decomp


def LWE(n,q,𝜎,s,t,m):
    """ Function to encrypt with Learning With Error (LWE)
    Params:
    @n    (int): security parameter (underlying lattice dimension)
    @q    (int): quotient modulus Zq
    @𝜎  (float): standard deviation of discrete Gaussian
    @s  (array): secret key
    @t    (int): message modulus
    @m  (array): message to cipher
    Returns:
    @a (array): drawn vector for cipher
    @b   (int): output of encryption
    """
    # draw uniformly random a
    a = draw_from_integer_vec(n=n,q=q)
    # draw discrete gaussian error
    e = draw_from_normal_vec(n=1,q=q,loc=0,scale=𝜎)
    # compute ratio
    delta = (q//t)
    # compute cifer
    b = (np.dot(a,s) + e + (delta*m)) % q
    return a,b


def inv_LWE(n,q,s,t,a,b):
    """ Function to decrypt with Learning With Error
    Params:
    @n    (int): security parameter (underlying lattice dimension)
    @q    (int): quotient modulus Zq
    @s  (array): secret key
    @t    (int): message modulus
    @a  (array): drawn vector for cipher
    @b    (int): output of encryption
    Returns:
    @m_decifer (int): decoded message
    """
    temp = (b-np.dot(a,s))[0] % q 
    return (np.round((t/q)*temp)%t).astype(int)
    

def RLWE(n,q,𝜎,s,t,m):
    """ Function to encrypt with Ring Learning With Error (RLWE)
    Params:
    @n    (int): security parameter (underlying lattice dimension)
    @q    (int): quotient modulus Zq
    @𝜎  (float): standard deviation of discrete Gaussian
    @s (poly1d): secret key
    @t    (int): message modulus
    @m (poly1d): message to cipher
    Returns:
    @a (poly1d): drawn vector for cipher
    @b (poly1d): output of encryption
    """
    # draw uniformly random a
    a = draw_from_integer(n=n,q=q)
    # draw discrete gaussian error
    e = draw_from_normal(n=n,q=q,loc=0,scale=𝜎)
    # compute ratio
    delta = (q//t)
    # compute cifer
    b = mod(poly=(a*s) + e + (delta*m),
            q=q,
            poly_modulus=np.poly1d([1] + ((n - 1) * [0]) + [1]))
    return a,b


def inv_RLWE(n,q,s,t,a,b):
    """ Function to decrypt with Ring Learning With Error (RLWE)
    Params:
    @n    (int): security parameter (underlying lattice dimension)
    @q    (int): quotient modulus Zq
    @s (poly1d): secret key
    @t    (int): message modulus
    @a (poly1d): drawn vector for cipher
    @b (poly1d): output of encryption
    Returns:
    @m_decifer (poly1d): decoded message
    """
    temp = mod(poly=b-(a*s),q=q,poly_modulus=np.poly1d([1] + ((n - 1) * [0]) + [1]))
    return  np.poly1d((np.round((t/q)*temp)%t).astype(int))


def RGSW(n,q,𝜎,s,t,B,m):
    """ Function to encrypt with RGSW
    Params:
    @n    (int): security parameter (underlying lattice dimension)
    @q    (int): quotient modulus Zq
    @𝜎  (float): standard deviation of discrete Gaussian
    @s (poly1d): secret key
    @t    (int): message modulus
    @B    (int): base for decomposition
    @m (poly1d): message to cipher
    Returns:
    @A (matrix (2k,2) of poly1d)
    """
    pol_mod = np.poly1d([1] + ((n - 1) * [0]) + [1])
    k = floor(log(q,B))
    # compute vector (1,B,B^2,...,B^{k-1})
    powers = B**(np.arange(k))
    # compute matrix G
    G = np.kron(a=np.eye(2),b=powers.reshape(-1,1))
    # Encode all 0 with RLWE
    A = np.empty(shape=((2*k,2)),dtype=object)
    for i in range(2*k):
        a,b = RLWE(n=n,q=q,𝜎=𝜎,s=s,t=t,m=np.poly1d(np.zeros(n+1)))
        A[i,0] = mod(poly=(a + G[i,0]*m),
                     q=q,poly_modulus=pol_mod)
        A[i,1] = mod(poly=(b + G[i,1]*m),
                     q=q,poly_modulus=pol_mod)
    return A


def RLWE_prod(n,q,𝜎,s,t,B,m0,m1):
    """ Function to encrypt RLWE(m0) x RGSW(m1)
    Params:
    @n       (int): security parameter (underlying lattice dimension)
    @q       (int): quotient modulus Zq
    @𝜎     (float): standard deviation of discrete Gaussian
    @s    (poly1d): secret key
    @t       (int): message modulus
    @B       (int): base for decomposition
    @m0   (poly1d): message to cipher
    @m1   (poly1d): message to cipher
    Returns:
    @prod0 (poly1d): drawn vector for cipher
    @prod1 (poly1d): output of encryption
    """
    # compute RLWE(m0)
    a0,b0 = RLWE(n=n,q=q,𝜎=𝜎,s=s,t=t,m=m0)
    
    # decomp (a,b) in base B
    a_decomp = base_decomp(poly=a0, q=q, B=B)
    b_decomp = base_decomp(poly=b0, q=q, B=B)

    # compute M = RGSW(m1)
    M = RGSW(n=n,q=q,𝜎=𝜎,s=s,t=t,B=B,m=m1)

    # concatenate to perform product
    ab_decomp = np.concatenate((a_decomp,b_decomp))
    
    # perform product
    prod = np.tensordot(a=ab_decomp.reshape(1,-1),b=M,axes=1)
    prod0 = mod(poly=prod[0,0],q=q,poly_modulus=np.poly1d([1] + ((n - 1) * [0]) + [1]))
    prod1 = mod(poly=prod[0,1],q=q,poly_modulus=np.poly1d([1] + ((n - 1) * [0]) + [1]))
    return prod0, prod1

def prod_ext(n,q,a,b,B,M):
    """ Function to perfom prod RLWE x RGSW
    Params:
    @n       (int): security parameter (underlying lattice dimension)
    @q       (int): quotient modulus Zq
    @a    (poly1d): drawn vector for cipher (from RLWE)
    @b    (poly1d): output of encryption    (from RLWE)
    @B       (int): base for decomposition
    @M    (poly1d): RGSW
    Returns:
    @prod0 (poly1d): drawn vector for cipher
    @prod1 (poly1d): output of encryption
    """
    # decomp (a,b)=RLWE in base B
    a_decomp = base_decomp(poly=a, q=q, B=B)
    b_decomp = base_decomp(poly=b, q=q, B=B)
    # concatenate to perform product
    ab_decomp = np.concatenate((a_decomp,b_decomp))
    # perform product
    prod = np.tensordot(a=ab_decomp.reshape(1,-1),b=M,axes=1)
    prod0 = mod(poly=prod[0,0],q=q,poly_modulus=np.poly1d([1] + (int(n - 1) * [0]) + [1]))
    prod1 = mod(poly=prod[0,1],q=q,poly_modulus=np.poly1d([1] + (int(n - 1) * [0]) + [1]))
    return prod0, prod1
