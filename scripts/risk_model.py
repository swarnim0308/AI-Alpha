import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def fit_pca(returns, num_factor_exposures, svd_solver):
    """
    Fit PCA model with returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    num_factor_exposures : int
        Number of factors for PCA
    svd_solver: str
        The solver to use for the PCA model

    Returns
    -------
    pca : PCA
        Model fit to returns
    """
    
    pca = PCA(n_components=num_factor_exposures, svd_solver=svd_solver,random_state = 1)
    
    return pca.fit(returns)

def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    """
    Get the factor betas from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    factor_beta_indices : 1 dimensional Ndarray
        Factor beta indices
    factor_beta_columns : 1 dimensional Ndarray
        Factor beta columns

    Returns
    -------
    factor_betas : DataFrame
        Factor betas
    """
    assert len(factor_beta_indices.shape) == 1
    assert len(factor_beta_columns.shape) == 1
    
    factor_betas = pd.DataFrame(data = pca.components_.T,index = factor_beta_indices,columns = factor_beta_columns)
    return factor_betas

def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    """
    Get the factor returns from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    returns : DataFrame
        Returns for each ticker and date
    factor_return_indices : 1 dimensional Ndarray
        Factor return indices
    factor_return_columns : 1 dimensional Ndarray
        Factor return columns

    Returns
    -------
    factor_returns : DataFrame
        Factor returns
    """
    assert len(factor_return_indices.shape) == 1
    assert len(factor_return_columns.shape) == 1
    
    factor_returns = pd.DataFrame(data = pca.transform(returns),index = factor_return_indices,columns = factor_return_columns)
    
    return factor_returns

def factor_cov_matrix(factor_returns, ann_factor = 252):
    """
    Get the factor covariance matrix

    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns
    ann_factor : int
        Annualization factor

    Returns
    -------
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    """
    
    factor_cov_matrix = ann_factor * np.diag(factor_returns.var(axis=0,ddof=1))
    
    return factor_cov_matrix

def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    """
    Get the idiosyncratic variance matrix

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    factor_returns : DataFrame
        Factor returns
    factor_betas : DataFrame
        Factor betas
    ann_factor : int
        Annualization factor

    Returns
    -------
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    """
    
    portfolio_var=pd.DataFrame(np.dot(factor_returns,factor_betas.T),returns.index,returns.columns)
    
    residuals=returns-portfolio_var
    
    idiosyncratic_var_matrix = pd.DataFrame(data = np.diag(ann_factor * np.var(residuals)),index = returns.columns,columns = returns.columns)
    
    return idiosyncratic_var_matrix

def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    """
    Get the idiosyncratic variance vector

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix

    Returns
    -------
    idiosyncratic_var_vector : DataFrame
        Idiosyncratic variance Vector
    """
    idiosyncratic_var_vector = pd.DataFrame(data = np.diag(idiosyncratic_var_matrix), index = idiosyncratic_var_matrix.index)
    return idiosyncratic_var_vector

def risk_modelling(returns,num_factor_exposures = 20):
	
	ann_factor = 252

	risk_model = {}
	
	# Fitting The PCA Model
	pca = fit_pca(returns, num_factor_exposures, 'full')

	# Calculating Factor Betas
	risk_model['factor_betas'] = factor_betas(pca, returns.columns.values, np.arange(num_factor_exposures))

	# Calculating Risk Factors Returns
	risk_model['factor_returns'] = factor_returns(pca,returns,returns.index,np.arange(num_factor_exposures))

	# Calculating Factor Covariance Matrix
	risk_model['factor_cov_matrix'] = factor_cov_matrix(risk_model['factor_returns'], ann_factor)

	# Calculating Idiosyncratic Variance Matrix
	risk_model['idiosyncratic_var_matrix'] = idiosyncratic_var_matrix(returns, risk_model['factor_returns'], risk_model['factor_betas'], ann_factor)

	# Calculating Idiosyncratic Variance Vector
	risk_model['idiosyncratic_var_vector'] = idiosyncratic_var_vector(returns, risk_model['idiosyncratic_var_matrix'])

	return risk_model
