# -*- coding: utf-8 -*-

"""
(Fused) Gromov-Wasserstein approximation methods (Spar-GW, SaGroW, EGW-based, and AE)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from ot.utils import check_random_state
from ot.bregman import sinkhorn
from ot.lp import emd, emd2_1d
from scipy import sparse
from scipy.sparse import csr_matrix
import torch


def gw_distance(C1, C2, loss_fun, T, M=False, alpha=0):
    """
    Compute the GW (or FGW) distance between (C_1, a) and (C_2, b) with a given transport plan T.
    
    Parameters
    ----------
    C1 : np.array or csr_matrix
        relation matrix of source samples
    C2 : np.array or csr_matrix
        relation matrix of target samples
    loss_fun : function, R \times R \mapsto R
        loss function used for the distance, the transport plan does not depend on the loss function
    T : np.array
        transport plan matrix
    M : np.array, optional
        feature distance matrix between source and target samples
    alpha : float, optional
        trade-off parameter between struction and feature information
    
    Returns
    -------
    gw_dist : float
        GW (or FGW) distance
    """
    if np.any(np.isnan(T)):
        return np.nan
    
    if loss_fun in ["square_loss", "kl_loss"]:
        a = np.sum(T, axis=1)
        b = np.sum(T, axis=0)
        if isinstance(C1, np.ndarray):
            constC, hC1, hC2 = init_matrix(C1, C2, a, b, loss_fun)
        else:
            constC, hC1, hC2 = init_matrix_sparse(C1, C2, a, b, loss_fun)
        gw_dist = gwloss(constC, hC1, hC2, T)
    
    else:
        nz_id = T.nonzero()
        Lam = 0
        for jj in range(len(nz_id[0])):
            i = nz_id[0][jj]
            k = nz_id[1][jj]
            Lam += np.sum(loss_fun(C1[i, np.newaxis, :, np.newaxis],
                                   C2[k, np.newaxis, np.newaxis, :]), axis=0) * T[i, k]
        
        gw_dist = np.sum(Lam * T)
        
    if alpha > 0:
        gw_dist = (1-alpha) * np.sum(M * T) + alpha * gw_dist
    
    return gw_dist
    


def spar_gw(C1, C2, a, b, loss_fun, nb_samples, epsilon, M=False, alpha=0,
            solver='entropy', ipot_iter=10,
            max_iter=100, stop_thr=1e-9, random_state=False, verbose=False):
    """
    The proposed Spar-GW (or Spar-FGW) algorithm.

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
    M : np.array, optional
        feature distance matrix between source and target samples
    alpha : float, optional
        trade-off param. between struction and feature information
    solver : string, optimal
        regularization term, choose between 'entropy', 'proximal', and 'inexact-proximal'
    ipot_iter : int, optional
        number of inner Sinkhorn iterations for 'inexact-proximal'
    max_iter : int, optional
        max number of iterations
    stop_thr : float, optional
        stop threshold on error
    random_state : int, optional
        fix the seed for reproducibility

    Returns
    -------
    T : np.array
        GW (or FGW) transport plan
    """
    prob = np.sqrt(np.outer(a, b))
    prob /= np.sum(prob)
    prob *= nb_samples
    prob[prob>1] = 1
    
    if random_state:
        torch.manual_seed(random_state)
    mask = torch.bernoulli(torch.from_numpy(prob)).numpy()
    
    T = np.outer(a, b)
    
    if loss_fun in ["square_loss", "kl_loss"]:
        if isinstance(C1, np.ndarray):
            constC, hC1, hC2 = init_matrix(C1, C2, a, b, loss_fun)
        else:
            constC, hC1, hC2 = init_matrix_sparse(C1, C2, a, b, loss_fun)

    for cpt in range(max_iter):
        
        if loss_fun in ["square_loss", "kl_loss"]:
            Lik = gwggrad(constC, hC1, hC2, T) / 2
            
        else:
            if isinstance(C1, np.ndarray) == False:
                C1 = C1.toarray()
                C2 = C2.toarray()
                
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
        
        if alpha > 0:
            Lik = (1-alpha) * M + alpha * Lik

        max_Lik = np.max(Lik)
        if max_Lik == 0:
            continue
        Lik /= max_Lik
        
        # Importance sparsification of the kernel matrix in Sinkhorn iterations.
        K = np.exp(-Lik/epsilon) 
        K_spar = np.zeros((len(a), len(b)))
        K_spar[mask!=0] = K[mask!=0]/prob[mask!=0]
        
        try:
            if solver == 'entropy':
                new_T = sinkhorn_plan(a, b, K_spar)
            elif solver == 'proximal':
                K_spar = K_spar * T
                new_T = sinkhorn_plan(a, b, K_spar)
            elif solver == 'inexact-proximal':
                new_T = ipot_plan(a, b, K_spar, ipot_iter)
        except (RuntimeWarning, UserWarning, ValueError, IndexError):
            print("Warning catched in Sinkhorn")
            T = np.nan
            break
        
        if np.any(np.isnan(new_T)):
            new_T = T
            print("Numerical errors in Sinkhorn")
            T = np.nan
            break

        
        if cpt % 10 == 0:
            change_T = np.linalg.norm(T - new_T)
            # change_T = np.mean((T - new_T) ** 2)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', '||T_n - T_{n+1}||') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, change_T))

            if change_T <= stop_thr:
                T = np.copy(new_T)
                break
            
        T = np.copy(new_T)

    return T


def sinkhorn_plan(a, b, K, max_iter=1000, stop_thr=1e-9):
    """
    Solve the entropic regularization optimal transport problem between a and b.
    
    Parameters
    ----------
    a : np.array
        distribution in the source space
    b : np.array
        distribution in the target space
    K : np.array
        kernel matrix
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

    err = 1.
    
    for ii in range(max_iter):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        u = a / Kv
        Ktu = K.T.dot(u)
        v = b / Ktu

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



def ipot_plan(a, b, K, ipot_iter=10, max_iter=100, stop_thr=1e-9):
    """
    Approximate the exact optimal transport plan between a and b
    using IPOT (Inexact Proximal Optimal Transport) method.
    
    Parameters
    ----------
    a : np.array
        distribution in the source space
    b : np.array
        distribution in the target space
    K : np.array
        kernel matrix
    ipot_iter : int, optional
        number of inner Sinkhorn iterations
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
    
    G_temp = np.outer(a, b)
    
    u = np.ones(dim_a) / dim_a
    v = np.ones(dim_b) / dim_b
    
    # if min(ns, nt) >= 200:
    #     K = sparse.csr_matrix(K)

    err = 1.
    
    for ii in range(max_iter):
        uprev = u
        vprev = v
        newK = K * G_temp
        
        for jj in range(ipot_iter):
            Kv = newK.dot(v)
            u = a / Kv
            Ktu = newK.T.dot(u)
            v = b / Ktu
        
        G_temp = np.expand_dims(u,axis=1) * newK * np.expand_dims(v,axis=0)
        
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

    if isinstance(newK, np.ndarray):
        G_temp = u[:, None] * newK * v[None, :]
    else:
        G_temp = u[:, None] * newK.toarray() * v[None, :]
    
    G_temp2 = np.zeros((ns, dim_b))
    G_temp2[id_row, ] = G_temp
    G = np.zeros((ns, nt))
    G[:, id_col] = G_temp2
    
    return G



# Code is copied from POT and modified for our setup.
def sampled_gw(C1, C2, a, b, loss_fun, nb_samples_grad, epsilon, M=False, alpha=0, learning_step=0.8, KL=False, 
               max_iter=100, con_loop=50, stop_thr=1e-9, verbose=False, random_state=False):
    """
    SaGroW algorithm, adapted for being compatible with FGW.
    """
    ns = a.shape[0]
    nt = b.shape[0]
    
    generator = check_random_state(random_state)

    T = np.outer(a, b)
    
    # continue_loop allows to stop the loop if there is several successive small modification of T.
    continue_loop = 0

    # The gradient of GW is more complex if the two matrices are not symmetric.
    C_are_symmetric = np.allclose(C1, C1.T, rtol=1e-10, atol=1e-10) and np.allclose(C2, C2.T, rtol=1e-10, atol=1e-10)

    for cpt in range(max_iter):
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
                
        if alpha > 0:
            Lik = (1-alpha) * M + alpha * Lik

        max_Lik = np.max(Lik)
        if max_Lik == 0:
            continue
        # This division by the max is here to facilitate the choice of epsilon.
        Lik /= max_Lik

        if KL == True:
            # Set to infinity all the numbers below exp(-200) to avoid log of 0.
            log_T = np.log(np.clip(T, np.exp(-200), 1))
            log_T = np.where(log_T == -200, -np.inf, log_T)
            Lik = Lik - epsilon * log_T
            try:
                new_T = sinkhorn(a=a, b=b, M=Lik, reg=epsilon)
            except (RuntimeWarning, UserWarning):
                print("Warning catched in Sinkhorn: Return last stable T")
                break
        
        else:
            try:
                new_T = sinkhorn(a=a, b=b, M=Lik, reg=epsilon)
            except (RuntimeWarning, UserWarning):
                print("Warning catched in Sinkhorn: Return last stable T")
                break
            new_T = (1 - learning_step) * T + learning_step * new_T
            
        if np.any(np.isnan(new_T)):
            new_T = T
            print("Numerical errors in Sinkhorn: Return last stable T")
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
def entropic_gw(C1, C2, a, b, loss_fun, epsilon, M=False, alpha=0, solver='entropy', 
                max_iter=100, stop_thr=1e-9, verbose=False):
    """
    EGW-based methods (EGW if solver='entropy' and epsilon>0; 
                       PGA-GW if solver='proximal' (or 'inexact-proximal') and epsilon>0; 
                       EMD-GW if epsilon=0), 
    adapted for being compatible with FGW.
    """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    T = np.outer(a, b)  # Initialization
    
    if loss_fun in ["square_loss", "kl_loss"]:
        constC, hC1, hC2 = init_matrix(C1, C2, a, b, loss_fun)

    cpt = 0
    err = 1.

    while (err > stop_thr and cpt < max_iter):

        Tprev = T

        # compute the gradient
        if loss_fun in ["square_loss", "kl_loss"]:
            tens = gwggrad(constC, hC1, hC2, T)
        else:
            tens = 2 * compute_L(C1=C1, C2=C2, loss_fun=loss_fun, T=T)
            
        if alpha > 0:
            tens = (1-alpha) * M + alpha * tens / 2

        if epsilon > 0:
            try:
                if solver == 'entropy':
                    m = np.max(tens)
                    T = sinkhorn(a, b, tens/m, epsilon)
                elif solver == 'proximal':
                    log_T = np.log(np.clip(T, np.exp(-200), 1))
                    log_T[log_T == -200] = -np.inf
                    tens = tens - epsilon * log_T
                    m = np.max(tens)
                    T = sinkhorn(a, b, tens/m, epsilon)
                elif solver == 'inexact-proximal':
                    m = np.max(tens)
                    K = np.exp(-tens/(m*epsilon))
                    T = ipot_plan(a, b, K)
            except:
                print("The method is not converged. Return last stable T. Nb iter : " + str(cpt))
                break
        else:
            try:
                T = emd(a, b, tens)
            except:
                print("The method is not converged. Return last stable T. Nb iter : " + str(cpt))
                break

        if cpt % 10 == 0:
            err = np.linalg.norm(T - Tprev)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    return T


def compute_L(C1, C2, loss_fun, T):
    
    s = 0
    for i in range(T.shape[0]):
        for k in range(T.shape[1]):
            s = s + np.sum(loss_fun(C1[i, np.newaxis, :, np.newaxis],
                            C2[k, np.newaxis, np.newaxis, :]), axis=0) * T[i, k]
    return s


def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
    
    if loss_fun == 'square_loss':
        def f1(a):
            return (a ** 2)
        def f2(b):
            return (b ** 2)
        def h1(a):
            return a
        def h2(b):
            return 2 * b
        
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a
        def f2(b):
            return b
        def h1(a):
            return a
        def h2(b):
            return np.log(b + 1e-15)

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def init_matrix_sparse(cost_s: csr_matrix, cost_t: csr_matrix,
                       p: np.ndarray, q: np.ndarray, loss_fun: str = 'square_loss'):
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    if loss_fun == 'square_loss':
        constC1 = np.repeat((cost_s.power(2) @ p.reshape(-1, 1)), n_t, axis=1)
        constC2 = np.repeat((cost_t.power(2) @ q.reshape(-1, 1)).T, n_s, axis=0)
        hC1 = cost_s
        hC2 = 2 * cost_t
        
    elif loss_fun == "kl_loss":
        constC1 = np.repeat(np.matmul(cost_s.multiply(np.log(cost_s + 1e-15)) - cost_s, p.reshape(-1, 1)), n_t, axis=1)
        constC2 = np.repeat((cost_t @ q.reshape(-1, 1)).T, n_s, axis=0)
        hC1 = cost_s
        hC2 = np.log(cost_t + 1e-15) 

    constC = constC1 + constC2

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):
    # A = -np.dot(hC1, T).dot(hC2.T)
    A = -hC1 @ T @ hC2.T
    tens = constC + A
    return tens


def gwloss(constC, hC1, hC2, T): 
    tens = tensor_product(constC, hC1, hC2, T)
    return np.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):  
    return 2 * tensor_product(constC, hC1, hC2, T) 


def define_loss_function(loss_func_name, GPU=False):
    
    if loss_func_name == "square_loss":
        def loss_fun(C1, C2):
            return (C1 - C2) ** 2

    elif loss_func_name == "1_loss":
        def loss_fun(C1, C2):
            return np.abs(C1 - C2)

    return loss_fun



def anchor_energy(C1, C2, a, b, p_order=2, method='1d', epsilon=1, max_iter=100, stop_thr=1e-6):
    """
    AE method.
    """
    n_s = C1.shape[0]
    n_t = C2.shape[0]
    
    ae_dist = 0
    
    if method == '1d':
        for i in range(n_s):
            for j in range(n_t):
                if p_order == 2:
                    ae_dist += a[i] * b[j] * emd2_1d(C1[i,], C2[:,j], a, b, metric='sqeuclidean')
                else:
                    ae_dist += a[i] * b[j] * emd2_1d(C1[i,], C2[:,j], a, b, metric='minkowski') 
                    
    return ae_dist
