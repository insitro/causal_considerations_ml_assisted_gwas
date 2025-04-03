import numpy as np
from scipy.stats import multivariate_normal
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr



def simulate_phenotypes_and_biomarkers_with_correlations(N, n, m, k, n_B, n_P, n_BP, p, sigma_B, sigma_P, sigma_BP, sigma_epsilon_B, sigma_epsilon_P):
    """
    Simulate phenotypes and biomarkers with specified levels of heritability and genetic correlation,
    and calculate genetic correlations for each biomarker, each phenotype, and each pair of biomarker/phenotype.

    Parameters:
    N (int): Number of samples
    n (int): Number of genetic variants.
    m (int): Number of biomarkers.
    k (int): Number of phenotypes.
    n_B (int): Number of causal variants for biomarkers.
    n_P (int): Number of causal variants for phenotypes.
    n_BP (int): Number of causal variants shared between biomarkers and phenotypes.
    p (float): Minor Allele Frequence.
    sigma_B (float): Average genetic variance of causal variants for biomarkers.
    sigma_P (float): Average genetic variance of causal variants for phenotypes.
    sigma_BP (float): Average covariance between the effect sizes of biomarkers and phenotypes due to shared causal variants.
    sigma_epsilon_B (float): Environmental variance for biomarkers.
    sigma_epsilon_P (float): Environmental variance for phenotypes.

    Returns:
    B (numpy.ndarray): Simulated biomarkers (N x m matrix).
    P (numpy.ndarray): Simulated phenotypes (N x k matrix).
    h2_B (float): Heritability of biomarkers.
    h2_P (float): Heritability of phenotypes.
    rho_B (numpy.ndarray): Genetic correlation matrix for biomarkers (m x m matrix).
    rho_P (numpy.ndarray): Genetic correlation matrix for phenotypes (k x k matrix).
    rho_BP (numpy.ndarray): Genetic correlation matrix between biomarkers and phenotypes (m x k matrix).
    G (numpy.ndarray): Genetic variant matrix (N x n matrix)
    causal_indices_B (numpy.ndarray): Indices of causal variants for biomarkers
    causal_indices_P (numpy.ndarray): Indices of causal variants for phenotypes 
    E_B (numpy.ndarray): Biomarker effect sizes
    E_P (numpy.ndarray): Phenotype effect sizes
    env_B (numpy.ndarray): Simulated environmental effect on biomarkers
    env_P (numpy.ndarray): Simulated environmental effect on phenotypes
    """
    # Generate genetic variants matrix for N samples and n variants
    G = np.random.binomial(2, p, (N, n))

    # Initialize phenotype and biomarker matrices
    B = np.zeros((N, m)) if m > 0 else None
    P = np.zeros((N, k)) if k > 0 else None

    # Initialize genetic effects matrices for biomarkers and phenotypes
    E_B = np.zeros((m, n)) if m > 0 else None
    E_P = np.zeros((k, n)) if k > 0 else None

    # Select causal indices for shared biomarkers and phenotypes
    causal_indices_shared = np.random.choice(n, n_BP, replace=False)

    # Select additional unique causal indices for biomarkers
    causal_indices_B_unique = np.random.choice(np.setdiff1d(np.arange(n), causal_indices_shared), n_B - n_BP, replace=False)

    # Select additional unique causal indices for phenotypes
    causal_indices_P_unique = np.random.choice(np.setdiff1d(np.arange(n),
                                                            np.concatenate((causal_indices_shared,
                                                                            causal_indices_B_unique))),
                                               n_P - n_BP, replace=False)

    # Combine the shared and unique causal indices for biomarkers and phenotypes
    causal_indices_B = np.concatenate((causal_indices_shared, causal_indices_B_unique))
    causal_indices_P = np.concatenate((causal_indices_shared, causal_indices_P_unique))
    
    env_B = np.random.normal(0, np.sqrt(sigma_epsilon_B), (N, m))
    env_P = np.random.normal(0, np.sqrt(sigma_epsilon_P), (N, k))

    # Generate causal effects for biomarkers
    if m > 0:
        for i in causal_indices_B:
            E_B[:, i] = multivariate_normal.rvs(mean=np.zeros(m), cov=sigma_B * np.eye(m))
        B = np.dot(G[:, causal_indices_B], E_B[:, causal_indices_B].T) + env_B
        
    # Generate causal effects for phenotypes
    if k > 0:
        for j in causal_indices_P:
            E_P[:, j] = multivariate_normal.rvs(mean=np.zeros(k), cov=sigma_P * np.eye(k))
        P = np.dot(G[:, causal_indices_P], E_P[:, causal_indices_P].T) + env_P
    
    # Calculate heritability for biomarkers and phenotypes
    h2_B = np.var(np.dot(G[:, causal_indices_B], E_B[:, causal_indices_B].T), axis=0) / np.var(B, axis=0) if m > 0 else None
    h2_P = np.var(np.dot(G[:, causal_indices_P], E_P[:, causal_indices_P].T), axis=0) / np.var(P, axis=0) if k > 0 else None

    # Calculate genetic correlation matrices
    if m > 0:
        rho_B = np.corrcoef(E_B)
    else:
        rho_B = None
        
    if k > 0:
        rho_P = np.corrcoef(E_P)
    else:
        rho_P = None


    
    # Genetic correlation matrix between biomarkers and phenotypes
    if m > 0 and k > 0:
        shared_indices = np.intersect1d(causal_indices_B, causal_indices_P)
        rho_BP = np.zeros((m, k))

        if shared_indices.size > 0:
            EB_shared = E_B[:, shared_indices]
            EP_shared = E_P[:, shared_indices]

            # Compute the correlations between the effects of biomarkers and phenotypes for shared causal variants
            for i in range(m):  # Loop through each biomarker
                for j in range(k):  # Loop through each phenotype
                    # Get the effects of the ith biomarker and jth phenotype for the shared indices
                    effects_B = EB_shared[i, :]
                    effects_P = EP_shared[j, :]

                    # Calculate the correlation between the effects
                    correlation = np.corrcoef(effects_B, effects_P)[0, 1]
                    rho_BP[i, j] = correlation

        else:
            rho_BP = np.empty((m, k)).fill(np.nan)
    else:
        rho_BP = np.empty((m, k)).fill(np.nan)


    return B, P, h2_B, h2_P, rho_B, rho_P, rho_BP, G, causal_indices_B, causal_indices_P, E_B, E_P, env_B, env_P


def simulate_downstream_phenotypes(biomarkers, correlations, min_env_variance):
    """
    Simulate downstream phenotypes from biomarkers with specified degree of environmental influence
    and assuming that the standard deviation of the phenotypes is 1.

    :param biomarkers: numpy array (N x m), where m = number of biomarkers, N = number of samples
    :param correlations: numpy array (k x m), where k is the number of downstream phenotypes,
                         containing the desired correlations between biomarkers and phenotypes
    :param min_env_variance: numpy array (k,), containing the minimum environmental variance for each phenotype
    :return: numpy array (N x k), simulated downstream phenotypes
    """
    N = biomarkers.shape[0]
    k, m = correlations.shape

    std_biomarkers = np.std(biomarkers, axis=0)

    # Construct the A matrix to reflect genetic influence
    A = correlations / std_biomarkers

    # Calculate genetic contribution to variance
    genetic_variance = np.sum((A ** 2) * (std_biomarkers ** 2), axis=1)
    
    # Calculate the minimum environmental variance needed to reach a phenotype variance of 1
    base_env_variance = 1 - genetic_variance
    
    # Ensure that environmental variance is at least the provided minimum
    corrected_env_variance = np.maximum(base_env_variance, min_env_variance)

    # Simulate environmental effects with the proper variance
    epsilon = np.random.multivariate_normal(np.zeros(k), np.diag(corrected_env_variance), N)

    # Generate downstream phenotypes
    P_downstream = biomarkers @ A.T + epsilon
    
    # Normalize the final phenotypes to have standard deviation of 1 (optional)
    P_downstream = (P_downstream - np.mean(P_downstream, axis=0)) / np.std(P_downstream, axis=0)

    return P_downstream




def run_gwas_return_metrics(phenotype, Genotype_matrix, causal_indices, effect_size=None):
    """
    Perform per variant linear regression based association between Phenotype and Genotype matrix.
    Then generate standard metrics including the correlation between true effect size and estimated effect size.

    Parameters: 
    phenotype: numpy array (N x 1), N = number of samples.
    Genotype_matrix: numpy array (N x m), m = number of SNPs.
    causal_indices: numpy array (m x 1). Each element is the index of the causal variant.
    effect_size: numpy array (m x 1), where each element is the true effect size for each variant. 

    Returns:
    P_values: numpy array (m x 1), p-values of each SNP
    Beta_estimates: numpy array (m x 1), estimated effect size of each SNP
    TPR: float, True Positive Rate after correction
    FPR: float, False Positive Rate after correction
    FDR: float, False Discovery Rate
    Effect_correlation: float, correlation between true effect size and GWAS estimated effect size
    """

    n, m = Genotype_matrix.shape
    P_values = np.zeros(m)
    Beta_estimates = np.zeros(m)

    # Perform the per variant linear regression
    for i in range(m):
        # Prepare the design matrix for linear regression
        X = sm.add_constant(Genotype_matrix[:, i])

        # Perform linear regression
        model = sm.OLS(phenotype, X).fit()
        
        # Store the coefficient (effect size) and p-value for the SNP
        Beta_estimates[i] = model.params[1]
        P_values[i] = model.pvalues[1]

    # Apply Bonferroni correction
    bonferroni_alpha = 0.05 / m
    significant_indices = P_values < bonferroni_alpha

    # Calculate TPR and FPR
    true_positives = significant_indices[causal_indices].sum()
    TPR = true_positives / len(causal_indices)

    true_negatives = (~significant_indices[~np.isin(range(m), causal_indices)]).sum()
    false_positives = significant_indices[~np.isin(range(m), causal_indices)].sum()
    FPR = false_positives / (false_positives + true_negatives)

    # Calculate FDR
    predicted_positives = significant_indices.sum()
    FDR = false_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate the correlation between estimated effect sizes and true effect sizes if effect_sizes are provided
    if effect_size is not None:
        Effect_correlation, _ = pearsonr(Beta_estimates, effect_size)
    else:
        Effect_correlation = None

    return P_values, Beta_estimates, TPR, FPR, Effect_correlation, FDR



def fit_regression(phenotype, biomarkers):
    """
    Fit a linear regression model: phenotype ~ biomarkers 

    Parameters: 
    phenotype: numpy array (N x 1), N = number of samples.
    biomarkers: numpy array (N x m), m = number of biomarkers.

    Returns:
    pheno_pred: numpy array (N x 1), predicted value of phenotype using biomarkers 
    """
    # Add a constant (intercept) to the biomarkers matrix
    biomarkers_with_intercept = sm.add_constant(biomarkers)

    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(phenotype, biomarkers_with_intercept)
    results = model.fit()
    
    # Make predictions using the biomarkers
    pheno_pred = results.predict(biomarkers_with_intercept)
    
    return pheno_pred



def simulate_downstream_phenotypes_with_direct_genetic_effects(N, G, biomarkers, correlations,
                                                              env_variance, n_pd, sigma_pd,
                                                              causal_indices_B):
    """
    Simulate downstream phenotypes from biomarkers and direct genetic effects with specified
    degree of environmental influence.

    Parameters:
    N (int): Number of samples
    G (numpy.ndarray): Genetic variant matrix (N x n matrix)
    biomarkers (numpy.ndarray): Simulated biomarkers (N x m matrix)
    correlations (numpy.ndarray): Desired correlations between biomarkers and phenotypes (k x m matrix)
    min_env_variance (numpy.ndarray): (k,) dim array containing the minimum environmental variance for each phenotype
    n_pd (int): Number of causal variants directly affecting downstream phenotypes
    sigma_pd (float): Variance of the direct genetic effect sizes for downstream phenotypes
    causal_indices_B (numpy.ndarray): Indices of causal variants for biomarkers

    Returns:
    P_downstream (numpy.ndarray): Simulated downstream phenotypes (N x k matrix)
    direct_h2 (numpy.ndarray): Direct heritability of downstream phenotypes (k,)
    indirect_h2 (numpy.ndarray): Indirect heritability of downstream phenotypes (k,)
    causal_indices_pd (numpy.ndarray): Indices of causal variants directly affecting downstream phenotypes
    """
    N, n = G.shape
    k, m = correlations.shape

    # Calculate the standard deviations of the biomarkers
    std_biomarkers = np.std(biomarkers, axis=0)

    # Construct the A matrix to reflect genetic influence
    A = correlations / std_biomarkers

    # Calculate genetic contribution to variance through biomarkers
    genetic_variance = np.sum((A ** 2) * (std_biomarkers ** 2), axis=1)

    # Calculate environmental variance needed to reach a phenotype variance of 1, given indirect effects
    base_env_variance = 1 - genetic_variance
    
    # Ensure that environmental variance is at least the given minimum
    corrected_env_variance = np.maximum(base_env_variance, env_variance)

    # Select causal variants for direct genetic effects on downstream phenotypes
    causal_indices_pd = np.random.choice(np.setdiff1d(np.arange(n), causal_indices_B), n_pd, replace=False)

    # Generate direct genetic effects for downstream phenotypes
    E_pd = np.zeros((n, k))
    for i in causal_indices_pd:
        E_pd[i] = multivariate_normal.rvs(mean=np.zeros(k), cov=sigma_pd * np.eye(k))

    # Calculate direct genetic effects
    direct_genetic_effects = G[:, causal_indices_pd].dot(E_pd[causal_indices_pd])

    # Calculate indirect genetic effects through biomarkers
    indirect_genetic_effects = biomarkers.dot(A.T)

    # Simulate environmental effects for each sample with corrected variance
    epsilon = np.random.multivariate_normal(np.zeros(k), np.diag(corrected_env_variance), N)

    # Calculate the downstream phenotypes incorporating direct and indirect genetic effects and environmental effects
    P_downstream = indirect_genetic_effects + direct_genetic_effects + epsilon

    # Calculate variances for heritability estimates
    var_direct_genetic = np.var(direct_genetic_effects, axis=0)
    var_indirect_genetic = np.var(indirect_genetic_effects, axis=0)
    var_total = np.var(P_downstream, axis=0)

    # Calculate heritabilities
    direct_h2 = var_direct_genetic / var_total
    indirect_h2 = var_indirect_genetic / var_total

    return P_downstream, direct_h2, indirect_h2, causal_indices_pd




    