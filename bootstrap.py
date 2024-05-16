'''
Date: May 6th 2022
Authors: 
Njaka ANDRIAMANDRATOMANANA, Elie CHEDEMAIL,
Adéchola KOUANDE, Rémi LELUC, Quyen NGUYEN
This file contains some functions to perform Bootstrap in FHE
'''
# List of functions
# 1.extract_LWE
# 2.key_switch
# 3.get_BK
# 4.get_BK_dec
# 5.accu_AP
# 6.accu_AP_dec
# 7.bootstrap_AP
# 8.bootstrap_AP_dec

# import libraries
import numpy as np
from tqdm import tqdm
from math import log,floor
from utils import mod, base_decomp, base_decomp_vec
from fhe import LWE
from fhe import RLWE, RGSW, prod_ext

# Function to extract cipher coeff_i from RLWE
def extract_RLWE(n,a,b,i_coeff):
    """ Function to extract cipher coeff_i from (a,b)=RLWE
    Params:
    @a    (poly1d): drawn vector for cipher
    @b    (poly1d): output of encryption
    @i_coeff (int): index of wanted encrypted coeff
    Returns:
    @a_i     (int):
    @coeff_b (int):
    """
    coeff_b = b.coef[-1-i_coeff]
    if len(a.coef)<n:
        a_vec = np.insert(a.coef,0,(n-len(a.coef))*[0])
    else:
        a_vec = a.coef
    a_i = np.roll(a=a_vec,shift=i_coeff+1)
    return a_i, np.array([coeff_b])


# Function to perform KeySwitch
def key_switch(n,q,B,𝜎,t,s,s_prime,a_prime,b_prime):
    """ Function to perform KeySwitch 
    from (a',b')=RLWE_{s'}(P) to (a,b)=RLWE_{s}(P) 
    Params:
    @n          (int): security parameter (underlying lattice dimension)
    @q          (int): quotient modulus Zq
    @B          (int): base for decomposition
    @𝜎        (float): standard deviation of discrete Gaussian
    @s       (poly1d): secret key
    @s_prime (poly1d): secret key
    @a_prime (poly1d): drawn vector for cipher
    @b_prime (poly1d): output of encryption
    Returns:
    @a    (poly1d): drawn vector for cipher
    @b    (poly1d): output of encryption
    """
    k = floor(log(q,B))
    # compute vector (1,B,B^2,...,B^{k-1})
    powers = B**(np.arange(k))
    # decomposition of a_prime
    a_prime_dec = base_decomp(poly=a_prime,q=q,B=B)
    # initialize matrix of KeySwitch values
    KS = np.empty((k,2),dtype=object)
    # fill KS matrix
    for j,Bj in enumerate(powers):
        temp = np.poly1d( int(Bj*s_prime) % q)
        # Encrypt the RLWE_s (B^j s')
        KS_ja, KS_jb = RLWE(n=n,q=q,𝜎=𝜎,s=s,t=t,m=temp)
        KS[j,0] = KS_ja
        KS[j,1] = KS_jb
    # perform product
    prod = np.tensordot(a=a_prime_dec.reshape(1,-1),b=KS,axes=1)
    
    pol_mod = np.poly1d([1] + ((n - 1) * [0]) + [1])
    prod0 = mod(poly=prod[0,0],q=q,poly_modulus=pol_mod)
    prod1 = mod(poly=prod[0,1],q=q,poly_modulus=pol_mod)
    
    a = mod(poly=-prod0,q=q,poly_modulus=pol_mod)
    b = mod(poly=b_prime - prod1,q=q,poly_modulus=pol_mod)
    return a,b

def get_BK(n,q,σ,t,B,s,s_prime):
    """ Compute matrix of Bootstrap Keys
    Params:
    @n       (int): security parameter (underlying lattice dimension)
    @q       (int): quotient modulus Zq
    @𝜎     (float): standard deviation of discrete Gaussian
    @s    (poly1d): secret key
    @t       (int): message modulus
    @B       (int): base for decomposition
    @s  (vector key): key for LWE
    @s_prime (poly1d key): secret key for bootstrap
    Returns:
    @BK (matrix of RGSW): Bootstrap Keys
    """
    k = int(round(log(q,B)))
    pol_mod = np.poly1d([1] + (int(n - 1) * [0]) + [1])
    # initialize Bootstrap Keys
    BK = np.empty(shape=(n,q,2*k,2), dtype=object)
    # compute Bootstrap Keys and fill matrix
    for i in range(n):
        for j in range(q):
            X_j_si = mod(poly=np.poly1d([1] + int(j*s[i]) * [0]),q=q,poly_modulus=pol_mod)
            BK[i,j] = RGSW(n=n, q=q, σ=σ, s=s_prime, t=t, B=B, m=X_j_si)
    return BK

def get_BK_dec(n,q,σ,t,B,Br,s,s_prime):
    """ Compute matrix of Bootstrap Keys with Decomposition Br
    Params:
    @n       (int): security parameter (underlying lattice dimension)
    @q       (int): quotient modulus Zq
    @𝜎     (float): standard deviation of discrete Gaussian
    @s    (poly1d): secret key
    @t       (int): message modulus
    @B       (int): base for decomposition in RGSW
    @Br      (int): base for decomposition in BK
    @s  (vector key): key for LWE
    @s_prime (poly1d key): secret key for bootstrap
    Returns:
    @BK (matrix of RGSW): Bootstrap Keys
    """
    k = floor(log(q,B))
    l = floor(log(q,Br))
    pol_mod = np.poly1d([1] + ((n - 1) * [0]) + [1])
    # initialize Bootstrap Keys
    BK = np.empty(shape=(n,l,Br,2*k,2), dtype=object)
    # compute Bootstrap Keys and fill matrix
    for i in tqdm(range(n)):
        for j in range(l):
            for v in range(Br): 
                X_v_j_si = mod(poly=np.poly1d([1] + int(v*j*s[i]) * [0]),q=q,poly_modulus=pol_mod)
                BK[i,j,v] = RGSW(n=n, q=q, σ=σ, s=s_prime, t=t, B=B, m=X_v_j_si)
    return BK

def accu_AP(n,q,B,a,b,w,BK):
    """ Function to perform Acc with AP
    Params:
    @n    (int): security parameter (underlying lattice dimension)
    @q    (int): quotient modulus Zq
    @a  (array): drawn vector for cipher
    @b    (int): output of encryption
    @w  (poly1d): polynomial well-chosen
    @BK (matrix of RGSW): Bootstrap Keys
    Returns:
    acc_a, acc_b = RLWE under s_prime
    """
    # perform trivial RLWE(0,X^{-b} w(X))
    acc_a = np.poly1d([0])
    if b[0]<n:
        monomial = np.poly1d([-1] + int(n - b[0]) * [0])
    else:
        monomial = np.poly1d([1] + int(2*n - b[0]) * [0])
    acc_b = mod(poly=(monomial*w),q=q,
                 poly_modulus=np.poly1d([1] + (int(n - 1) * [0]) + [1]))
    # perform recursive blind rotations
    for i in range(n):
        # perform external prod RLWE x RGSW
        acc_a, acc_b = prod_ext(n=n,q=q,a=acc_a,b=acc_b,
                                B=B,M=BK[i,a[i]])
    return acc_a, acc_b

def accu_AP_dec(n,q,B,Br,a,b,w,BK):
    """ Function to perform Acc with AP with Decomposition Br
    Params:
    @n    (int): security parameter (underlying lattice dimension)
    @q    (int): quotient modulus Zq
    @a  (array): drawn vector for cipher
    @b    (int): output of encryption
    @w  (poly1d): polynomial well-chosen
    @BK (matrix of RGSW): Bootstrap Keys
    Returns:
    acc_a, acc_b = RLWE under s_prime
    """
    l = floor(log(q,Br))
    # perform trivial RLWE(0,X^{-b} w(X))
    acc_a = np.poly1d([0])
    if b[0]<n:
        monomial = np.poly1d([-1] + int(n - b[0]) * [0])
    else:
        monomial = np.poly1d([1] + int(2*n - b[0]) * [0])
    acc_b = mod(poly=(monomial*w),q=q,
                poly_modulus=np.poly1d([1] + (int(n - 1) * [0]) + [1]))
    # perform recursive blind rotations
    for i in range(n):
        a_i_dec = base_decomp_vec(vec=a[i],q=q,Br=Br)
        for j in range(l):
            # perform external prod RLWE x RGSW
            acc_a, acc_b = prod_ext(n=n,q=q,a=acc_a,b=acc_b,
                                    B=B,M=BK[i,j,a_i_dec[j]])
    return acc_a, acc_b


def bootstrap_AP(n,q,σ,t,B,a,b,w,s,s_prime):
    """ Perform Bootstrap AP
    Params:
    @n       (int): security parameter (underlying lattice dimension)
    @q       (int): quotient modulus Zq
    @𝜎     (float): standard deviation of discrete Gaussian
    @s    (poly1d): secret key
    @t       (int): message modulus
    @B       (int): base for decomposition
    @s  (vector key): key for LWE
    @s_prime (poly1d key): secret key for bootstrap
    Returns:
    (a0,b0) = LWE_s with small errors
    """
    # get matrix of Bootstrap Keys
    BK = get_BK(n=n,q=q,σ=σ,t=t,B=B,s=s,s_prime=s_prime)
    # Step 1. ACC-part to get RLWE under s_prime
    a_prime, b_prime = accu_AP(n=n,q=q,B=B,a=a,b=b,w=w,BK=BK)
    # Step 2. Perform KeySwitch RLWE_{s'} --> RLWE_{s}
    a_poly, b_poly = key_switch(n=n,q=q,B=B,𝜎=𝜎,s=s,
                                s_prime=s_prime,a_prime=a_prime,b_prime=b_prime)
    # Step 3. Extract cipher LWE of coeff_0(RLWE_{s})
    a_0, b_0 = extract_RLWE(a=a_poly,b=poly,i_coeff=0)
    # return new cipher LWE
    return a_0, b_0

def bootstrap_AP_dec(n,q,σ,t,B,Br,a,b,w,s,s_prime):
    """ Perform Bootstrap AP with Decomposition Br
    Params:
    @n       (int): security parameter (underlying lattice dimension)
    @q       (int): quotient modulus Zq
    @𝜎     (float): standard deviation of discrete Gaussian
    @s    (poly1d): secret key
    @t       (int): message modulus
    @B       (int): base for decomposition
    @s  (vector key): key for LWE
    @s_prime (poly1d key): secret key for bootstrap
    Returns:
    (a0,b0) = LWE_s with small errors
    """
    # get matrix of Bootstrap Keys
    BK = get_BK_dec(n=n,q=q,σ=σ,t=t,B=B,Br=Br,s=s,s_prime=s_prime)
    # Step 1. ACC-part to get RLWE under s_prime
    a_prime, b_prime = accu_AP_dec(n=n,q=q,B=B,Br=Br,a=a,b=b,w=w,BK=BK)
    # Step 2. Perform KeySwitch RLWE_{s'} --> RLWE_{s}
    a_poly, b_poly = key_switch(n=n,q=q,B=B,𝜎=𝜎,s=s,
                                s_prime=s_prime,a_prime=a_prime,b_prime=b_prime)
    # Step 3. Extract cipher LWE of coeff_0(RLWE_{s})
    a_0, b_0 = extract_RLWE(a=a_poly,b=poly,i_coeff=0)
    # return new cipher LWE
    return a_0, b_0

