# -*- coding: utf-8 -*-

"""
Unbalanced Gromov-Wasserstein approximation methods (Spar-UGW, SaGroW, EUGW, and PGA-UGW)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from ot.utils import check_random_state
from scipy import sparse
import torch
from methods.gromov_funcs import init_matrix, init_matrix_sparse, gwggrad, gwloss, compute_L


def quad_kl_div(pi, ref):
    """
    Compute the quadratic KL divergence (KL^otimes(pi | ref))
    """
    massp = np.sum(pi)
    div = (
            2 * massp * np.sum(pi * np.log(pi/ref + 1e-20))
            - massp ** 2
            + np.sum(ref) ** 2
    )
    return div


def kl_div(pi, ref):
    """
    Compute the KL divergence (KL(pi | ref))
    """
    div = np.sum(pi * np.log(pi/ref + 1e-20)) - np.sum(pi) + np.sum(ref)
    return div


def ugw_distance(C1, C2, loss_fun, T, a, b, lam):
    """
    Compute the UGW distance between (C_1, a) and (C_2, b) with a given transport plan T.
    
    Parameters
    ----------
    C1 : np.array
        relation matrix of source samples
    C2 : np.array
        relation matrix of target samples
    loss_fun : function, R \times R \mapsto R
        loss function used for the distance, the transport plan does not depend on the loss function
    T : np.array
        transport plan matrix
    a : np.array
        distribution in the source space
    b : np.array
        distribution in the target space
    lam : float
        marginal relaxition parameter
    
    Returns
    -------
    gw_dist : float
        UGW distance
    """
    if np.any(np.isnan(T)):
        return np.nan
        
    if loss_fun in ["square_loss", "kl_loss"]:
        constC, hC1, hC2 = init_matrix(C1, C2, a, b, loss_fun)
        ugw_dist = gwloss(constC, hC1, hC2, T)
    
    else:
        nz_id = T.nonzero()
        Lam = 0
        for jj in range(len(nz_id[0])):
            i = nz_id[0][jj]
            k = nz_id[1][jj]
            Lam += np.sum(loss_fun(C1[i, np.newaxis, :, np.newaxis],
                                   C2[k, np.newaxis, np.newaxis, :]), axis=0) * T[i, k]
        
        ugw_dist = np.sum(Lam * T)
        
    ugw_dist += lam * quad_kl_div(np.sum(T, axis=1), a)
    ugw_dist += lam * quad_kl_div(np.sum(T, axis=0), b)
    
    return ugw_dist



def spar_ugw(C1, C2, a, b, loss_fun, nb_samples, epsilon, lam, 
             max_iter=100, con_loop=0, stop_thr=1e-9, random_state=False):
    """
    The proposed Spar-UGW algorithm.

    Parameters
    ----------
    C1 : np.array or csr_matrix
        relation matrix of source samples
    C2 : np.array or csr_matrix
        relation matrix of target samples
    a : np.array
        distribution in the source space
    b : np.array
        distribution in the target space
    loss_fun : function, R \times R \mapsto R
        loss function
    nb_samples : int
        number of selected elements
    epsilon : float
        regularization parameter
    lam : float
        marginal relaxition parameter
    max_iter : int, optional
        max number of iterations
    con_loop : int, optional
        max number of low modifications of T
    stop_thr : float, optional
        stop threshold on error
    random_state : int, optional
        fix the seed for reproducibility

    Returns
    -------
    T : np.array
        UGW transport plan
    """
    
    T = np.outer(a, b) / np.sqrt(np.sum(a) * np.sum(b))
    
    if loss_fun in ["square_loss", "kl_loss"]:
        if isinstance(C1, np.ndarray):
            constC, hC1, hC2 = init_matrix(C1, C2, a, b, loss_fun)
        else:
            constC, hC1, hC2 = init_matrix_sparse(C1, C2, a, b, loss_fun)

        Lik = gwggrad(constC, hC1, hC2, T) / 2
        
    else:
        try:
            Lik = np.sum(loss_fun(C1[:, :, np.newaxis, np.newaxis], C2[np.newaxis, np.newaxis, :, :])
                         * T[:, np.newaxis, :, np.newaxis], axis=(0, 2))
        
        except (MemoryError):
            Lik = 0
            index0 = range(a.shape[0])
            for i, index0_i in enumerate(index0):
                Lik += np.sum(loss_fun(
                    C1[[index0[i]] * b.shape[0], :][:, :, None],
                    C2[:, None, :]
                    ) * (T[index0_i, :])[:, None, None], 
                    axis = 0)
    
    Lik += (
            lam * np.sum(np.sum(T, axis=1) * np.log(np.sum(T, axis=1) / a + 1e-20))
            + lam * np.sum(np.sum(T, axis=0) * np.log(np.sum(T, axis=0) / b + 1e-20))
        )
    Lik /= np.max(Lik)
    
    K = np.exp(-Lik/epsilon) * T
    scale1 = epsilon / (2 * lam + epsilon)
    scale2 = lam / (2 * lam + epsilon)
    prob = K**scale1 * np.outer(a, b)**scale2
    prob /= np.sum(prob)
    prob *= nb_samples
    prob[prob>1] = 1
    
    if random_state:
        torch.manual_seed(random_state)
    mask = torch.bernoulli(torch.from_numpy(prob)).numpy()

    continue_loop = 0
    
    for cpt in range(max_iter):
        massT = np.sum(T)
        eps_temp = epsilon * massT
        lam_temp = lam * massT
        
        if cpt > 0:
            if loss_fun in ["square_loss", "kl_loss"]:
                Lik = gwggrad(constC, hC1, hC2, T) / 2
                
            else:
                new_mask = mask * (T!=0)
                index0 = np.where(np.sum(new_mask, axis=1)!=0)[0]
                
                Lik = 0
                for i, index0_i in enumerate(index0):
                    index1 = np.where(new_mask[index0_i, ]!=0)[0]
                    
                    Lik += np.sum(loss_fun(
                        C1[[index0[i]] * len(index1), :][:, :, None],
                        C2[index1, :][:, None, :]
                        ) * (T[index0_i, index1])[:, None, None], 
                        axis = 0)
            
            Lik += (
                lam * np.sum(np.sum(T, axis=1) * np.log(np.sum(T, axis=1) / a + 1e-20))
                + lam * np.sum(np.sum(T, axis=0) * np.log(np.sum(T, axis=0) / b + 1e-20))
            )

        # Importance sparsification of the kernel matrix in Sinkhorn iterations.
        K = np.exp(-Lik/eps_temp) * T
        K_spar = np.zeros((len(a), len(b)))
        K_spar[mask!=0] = K[mask!=0]/prob[mask!=0] 

        try:
            new_T = sinkhorn_plan_unbalanced(a, b, K_spar, eps_temp, lam_temp)
            new_T = np.sqrt(massT/np.sum(new_T)) * new_T
        except (RuntimeWarning, UserWarning, ValueError, IndexError):
            print("Warning catched in Sinkhorn: Return last stable T")
            T = np.nan
            break
        
        if np.any(np.isnan(new_T)):
            T = np.nan
            print("Numerical errors in Sinkhorn: Return last stable T")
            break
        
        change_T = np.mean((T - new_T) ** 2)
        
        if change_T <= stop_thr:
            continue_loop += 1
            if continue_loop > con_loop:
                T = np.copy(new_T)
                break
        else:
            continue_loop = 0

        T = np.copy(new_T)

    return T


def sinkhorn_plan_unbalanced(a, b, K, reg, reg_kl, max_iter=1000, stop_thr=1e-9):
    """
    Solve the entropic regularization unbalanced optimal transport problem between a and b.
    
    Parameters
    ----------
    a : np.array
        distribution in the source space
    b : np.array
        distribution in the target space
    K : np.array
        kernel matrix
    reg : float
        regularization parameter
    reg_kl : float
        marginal relaxition parameter
    max_iter : int, optional
        max number of iterations
    stop_thr : float, optional
        stop threshold on error

    Returns
    -------
    G : np.array
        transport plan
    """
    ns, nt = K.shape
    
    id_row = np.squeeze(np.asarray(np.where(np.sum(K, axis=1) != 0)))
    id_col = np.squeeze(np.asarray(np.where(np.sum(K, axis=0) != 0)))
    K = K[id_row, ][:, id_col]
    a = a[id_row]
    b = b[id_col]

    dim_a, dim_b = K.shape

    u = np.ones(dim_a) / dim_a
    v = np.ones(dim_b) / dim_b
    
    if min(ns, nt) >= 200:
        K = sparse.csr_matrix(K)
    
    fi = reg_kl / (reg_kl + reg)

    err = 1.
    
    for ii in range(max_iter):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        u = (a / Kv + 1e-30) ** fi 
        Ktu = K.T.dot(u)
        v = (b / Ktu + 1e-30) ** fi

        if (np.any(Ktu == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision, come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
            err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
            err = 0.5 * (err_u + err_v)

            if err < stop_thr:
                break

    if isinstance(K, np.ndarray):
        G_temp = u[:, None] * K * v[None, :]
    else:
        G_temp = u[:, None] * K.toarray() * v[None, :]
    
    G_temp2 = np.zeros((ns, dim_b))
    G_temp2[id_row, ] = G_temp
    G = np.zeros((ns, nt))
    G[:, id_col] = G_temp2
    
    return G



# Code is copied from POT and modified for our setup.
def sampled_ugw(C1, C2, a, b, loss_fun, nb_samples_grad, epsilon, lam, learning_step=0.8, KL=False, 
                max_iter=100, con_loop=50, stop_thr=1e-9, verbose=False, random_state=False):
    """
    SaGroW algorithm, adapted for UGW.
    """
    ns = a.shape[0]
    nt = b.shape[0]
    
    generator = check_random_state(random_state)

    T = np.outer(a, b) / np.sqrt(np.sum(a) * np.sum(b))
    
    # continue_loop allows to stop the loop if there is several successive small modification of T.
    continue_loop = 0

    # The gradient of GW is more complex if the two matrices are not symmetric.
    C_are_symmetric = np.allclose(C1, C1.T, rtol=1e-10, atol=1e-10) and np.allclose(C2, C2.T, rtol=1e-10, atol=1e-10)

    for cpt in range(max_iter):
        
        massT = np.sum(T)
        eps_temp = epsilon * massT
        lam_temp = lam * massT
        
        prob = T/np.sum(T)
        mask = np.zeros(ns*nt)
        if random_state:
            np.random.seed(random_state)
        if np.any(np.isnan(prob)):
            break
        id = np.random.choice(
            range(ns*nt), 
            nb_samples_grad, 
            p=prob.flatten(order='F'), 
            replace=True)
        mask[id] = 1
        mask = mask.reshape(ns, nt, order='F')

        index0 = np.where(np.sum(mask, axis=1)!=0)[0]

        Lik = 0

        for i, index0_i in enumerate(index0):
            index1 = np.where(mask[index0_i,]!=0)[0]
            
            # If the matrices C are not symmetric, the gradient has 2 terms, thus the term is chosen randomly.
            if (not C_are_symmetric) and generator.rand(1) > 0.5:
                Lik += np.mean(loss_fun(
                    C1[:, [index0[i]] * len(index1)][:, None, :],
                    C2[:, index1][None, :, :]
                ), axis=2)
            else:
                Lik += np.mean(loss_fun(
                    C1[[index0[i]] * len(index1), :][:, :, None],
                    C2[index1, :][:, None, :]
                ), axis=0)
        
        Lik += (
            lam * np.sum(np.sum(T, axis=1) * np.log(np.sum(T, axis=1) / a + 1e-20))
            + lam * np.sum(np.sum(T, axis=0) * np.log(np.sum(T, axis=0) / b + 1e-20))
        )

        max_Lik = np.max(Lik)
        if max_Lik == 0:
            continue
        # This division by the max is here to facilitate the choice of epsilon.
        Lik /= max_Lik

        if KL == True:
            K = np.exp(-Lik/eps_temp) * T
            try:
                new_T = sinkhorn_plan_unbalanced(a, b, K, eps_temp, lam_temp)
                new_T = np.sqrt(massT/np.sum(new_T)) * new_T
            except (RuntimeWarning, UserWarning, ValueError, IndexError):
                print("Warning catched in Sinkhorn: Return last stable T")
                T = np.nan
                break
        
        else:
            K = np.exp(-Lik/eps_temp)
            try:
                new_T = sinkhorn_plan_unbalanced(a, b, K, eps_temp, lam_temp)
                new_T = np.sqrt(massT/np.sum(new_T)) * new_T
            except (RuntimeWarning, UserWarning, ValueError, IndexError):
                print("Warning catched in Sinkhorn: Return last stable T")
                T = np.nan
                break
            new_T = (1 - learning_step) * T + learning_step * new_T
            
        if np.any(np.isnan(new_T)):
            print("Numerical errors in Sinkhorn: Return last stable T")
            T = np.nan
            break

        change_T = np.mean((T - new_T) ** 2)
        
        if change_T <= stop_thr:
            continue_loop += 1
            if continue_loop > con_loop:  # Number max of low modifications of T
                T = np.copy(new_T)
                break
        else:
            continue_loop = 0

        if verbose and cpt % 10 == 0:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format('It.', '||T_n - T_{n+1}||') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, change_T))
            
        T = np.copy(new_T)

    return T
    


# Code is copied from github: Hv0nnus/Sampled-Gromov-Wasserstein/GROMOV_personal.py and modified for our setup.
def entropic_ugw(C1, C2, a, b, loss_fun, epsilon, lam, 
                 max_iter=100, stop_thr=1e-9, KL=False, quad=False, verbose=False):
    """
    EUGW-based methods (EUGW if KL=False and quad=True, and PGA-UGW if KL=True).
    """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    T = np.outer(a, b) / np.sqrt(np.sum(a) * np.sum(b))  # Initialization
    
    if loss_fun in ["square_loss", "kl_loss"]:
        constC, hC1, hC2 = init_matrix(C1, C2, a, b, loss_fun)

    cpt = 0
    err = 1.

    while (err > stop_thr and cpt < max_iter):

        Tprev = T
        massTprev = np.sum(T)
        eps_temp = epsilon * massTprev
        lam_temp = lam * massTprev

        # compute the gradient
        if loss_fun in ["square_loss", "kl_loss"]:
            tens = gwggrad(constC, hC1, hC2, T) / 2
        else:
            tens = compute_L(C1=C1, C2=C2, loss_fun=loss_fun, T=T)
        
        tens += (
            lam * np.sum(np.sum(T, axis=1) * np.log(np.sum(T, axis=1) / a + 1e-20))
            + lam * np.sum(np.sum(T, axis=0) * np.log(np.sum(T, axis=0) / b + 1e-20))
            )
        
        max_tens = np.max(tens)

        if KL:
            K = np.exp(-(tens/max_tens)/eps_temp) * T
            
        elif quad:
            tens += epsilon * np.sum(T * np.log(T / np.outer(a, b) + 1e-20))
            max_tens = np.max(tens)
            K = np.exp(-(tens/max_tens)/eps_temp)
            
        else: 
            K = np.exp(-(tens/max_tens)/eps_temp)

        try:
            T = sinkhorn_plan_unbalanced(a, b, K, eps_temp, lam_temp)
            T = np.sqrt(massTprev/np.sum(T)) * T
        except:
            print("The method is not converged. Return last stable T. Nb iter : " + str(cpt))
            break


        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all the 10th iterations
            err = np.linalg.norm(T - Tprev)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    return T
