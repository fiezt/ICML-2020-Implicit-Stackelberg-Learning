from GameGradients import build_game_gradient, build_game_jacobian
from ComputationalTools import JacobianVectorProduct, SchurComplement, SchurComplement2
import scipy.sparse.linalg
import numpy as np
from torch import autograd

def calc_game_eigs(fs, xs, regularization=0, tol_gmres=1e-6, k=3, precise=False):
    
    G_loss, D_loss = fs
    G, D = xs
    Dg, Dd = build_game_gradient([G_loss, D_loss], [G, D])
    AA, BB, CC, DD, JJ = build_game_jacobian([Dg, Dd], [G, D])
    DD_reg = JacobianVectorProduct(Dd, list(D.parameters()), regularization)
    
    calc_eigs = lambda F: np.hstack((scipy.sparse.linalg.eigs(F, k=k, which='SR')[0], scipy.sparse.linalg.eigs(F, k=k, which='LR')[0]))
     
    A_eigs = calc_eigs(AA)
    D_eigs = calc_eigs(DD)
    D_reg_eigs = calc_eigs(DD_reg)
    J_eigs = calc_eigs(JJ)

    SC_reg = SchurComplement(AA, BB, CC, DD_reg, tol_gmres=tol_gmres, precise=precise)
    SC_reg2 = SchurComplement2(AA, BB, CC, DD, DD_reg, tol_gmres=tol_gmres, precise=precise)
    SC_reg_eigs = calc_eigs(SC_reg)
    SC_reg_eigs2 = calc_eigs(SC_reg2)

    return A_eigs, D_eigs, D_reg_eigs, J_eigs, SC_reg_eigs, SC_reg_eigs2



def calc_full_game_eigs(fs, xs, regularization=0):
    
    G_loss, D_loss = fs
    G, D = xs
    
    Dg, Dd = build_game_gradient([G_loss, D_loss], [G, D])
    AA, BB, CC, DD, JJ = build_game_jacobian([Dg, Dd], [G, D])
    
    DD_reg = JacobianVectorProduct(Dd, list(D.parameters()), regularization)

    DD_reg_mat = DD_reg.matmat(np.eye(DD_reg.shape[0], DD_reg.shape[0]))
    DD_reg_mat = DD_reg_mat.reshape(DD_reg.shape[0], DD_reg.shape[1])
    DD_reg_inv_mat = np.linalg.inv(DD_reg_mat)
    
    Dd_g = autograd.grad(G_loss, D.parameters(), create_graph=True)
    Ddd_g = JacobianVectorProduct(Dd_g, list(D.parameters()))
    
    Ddd_g_mat = Ddd_g.matmat(np.eye(Ddd_g.shape[0], Ddd_g.shape[1]))
    Ddd_g_mat = Ddd_g_mat.reshape(Ddd_g.shape[0], Ddd_g.shape[1])
    
    JJ_mat = JJ.matmat(np.eye(JJ.shape[0], JJ.shape[0]))
    JJ_mat = JJ_mat.reshape(JJ.shape[0], JJ.shape[1])
    
    AA_mat = JJ_mat[:AA.shape[0], :AA.shape[1]]
    BB_mat = JJ_mat[:AA.shape[0], AA.shape[1]:]
    CC_mat = JJ_mat[AA.shape[0]:, :AA.shape[1]]
    DD_mat = JJ_mat[AA.shape[0]:, AA.shape[1]:]
    

    SC_reg_mat = AA_mat - BB_mat@DD_reg_inv_mat@CC_mat
    SC_reg_mat2 = AA_mat - 2*BB_mat@DD_reg_inv_mat@CC_mat + CC_mat.T@DD_reg_inv_mat@Ddd_g_mat@DD_reg_inv_mat@CC_mat
    
    
    A_eigs = np.sort(np.real(np.linalg.eigvals(AA_mat)))
    D_eigs = np.sort(np.real(np.linalg.eigvals(DD_mat)))
    D_reg_eigs = np.sort(np.real(np.linalg.eigvals(DD_reg_mat)))
    J_eigs = np.sort(np.real(np.linalg.eigvals(JJ_mat)))
    SC_reg_eigs = np.sort(np.real(np.linalg.eigvals(SC_reg_mat)))
    SC_reg_eigs2 = np.sort(np.real(np.linalg.eigvals(SC_reg_mat2)))
    
    return A_eigs, D_eigs, D_reg_eigs, J_eigs, SC_reg_eigs, SC_reg_eigs2