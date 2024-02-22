import feather

import numpy as np
import pandas as pd 

from scipy import interpolate
import scipy.special as special

from questvar import utils

# GLOBAL VARIABLES
COLS = [
    "N1", "N2", "log2FC", 
    "df_p", "df_adjp", 
    "eq_lp", "eq_ladjp", 
    "eq_up", "eq_uadjp",
    "eq_p", "eq_adjp",
    "comb_p", "comb_adjp", 
    "log10(pval)", "log10(adj_pval)",
    "Status"
]

def printParams(
        combs_list: list[tuple[str, str]],
        pThr: float=0.05,
        dfThr: float=1,
        eqThr: float=0.5, 
        cvThr: float=75,
        is_log2: bool=False,
        equalVar: bool=True, 
        correction: str='fdr',
        is_paired: bool=False,
         
    ):
    """
        Prints the test parameters for the given test configuration.
    """
    print("Test Parameters:")
    print(f"  - p-value threshold: {pThr}")
    print(f"  - Equal variance: {equalVar}")
    print(f"  - Use Paired test: {is_paired}")
    print(f"  - Data in log2 Scale: {is_log2}")
    print(f"  - Correction method: {correction}")
    print(f"  - Filter higher than CV%: {cvThr}")
    print(f"  - Difference lfc boundary: (-{dfThr}, {dfThr})")
    print(f"  - Equivalence lfc boundary: (-{eqThr}, {eqThr})")
    print("")
    if len(combs_list) == 1:
        print(f"Single test for {combs_list[0][0]} vs {combs_list[0][1]}")
    elif len(combs_list) > 1 and len(combs_list) < 10:
        print(f"Multiple tests for:")
        for i, comb in enumerate(combs_list):
            print(f"  - {i+1}) {comb[0]} vs {comb[1]}")
    else:
        print(f"Multiple tests for {len(combs_list)} combinations")    
    

def _ttest_finish(
        df: np.ndarray, 
        t: np.ndarray, 
        alternative: str
    ):
    """
        Common code between all 3 t-test functions.
    """

    # Calculate p-value based on alternative hypothesis
    if alternative == 'less':
        pval = special.stdtr(df, t)
    elif alternative == 'greater':
        pval = special.stdtr(df, -t)
    elif alternative == 'two-sided':
        pval = special.stdtr(df, -np.abs(t))*2
    else:
        raise ValueError(
            "alternative must be "
            "'less', 'greater' or 'two-sided'"
        )
    # Return t-statistics and p-value
    if t.ndim == 0:
        t = t[()]
    if pval.ndim == 0:
        pval = pval[()]

    return t, pval

def _ttest_CI(
        df: np.ndarray,
        t: np.ndarray,
    ):
    """
        Calculate the confidence intervals for tests.
        Trying to improve the performance (WIP)
    """
    pass

def ttest_ind_with_na(
        m1: np.ndarray,
        m2: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray, 
        equal_var: bool=True,
        alternative: str='two-sided'
    ):
    
    """
        The ttest_ind function from scipy.stats 
        but with the ability to handle missing values.
    """
    # If the samples have equal variances
    if equal_var:
        # Calculate the degrees of freedom
        df = ((n1 + n2) - 2)
        # Calculate the pooled variance
        svar = ((n1-1)*v1 + (n2-1)*v2) / df
        # Calculate the denominator
        denom = np.sqrt(svar*(1.0/n1 + 1.0/n2))
    # If the variances are not equal
    else:
        # Calculate the degrees of freedom
        vn1 = v1/n1
        vn2 = v2/n2
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))
            
        # If df is undefined, variances are zero.
        # It doesn't matter what df is as long as it is not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)
    
    # Calculate t-statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (m1-m2) / denom
        
    return _ttest_finish(df, t, alternative)

def run_unpaired(
        S1_arr: np.ndarray,
        S2_arr: np.ndarray,
        pThr: float=0.05,
        dfThr: float=1,
        eqThr: float=0.5, 
        equalVar: bool=True, 
        correction: str='fdr', 
    ):
    """
        Running combined t-test and tost for the 
        given 2 samples on an unpaired configuration.
    """  

    # Calculate sample sizes for each protein in each sample
    n1 = S1_arr.shape[1] - np.sum(np.isnan(S1_arr), axis=1)
    n2 = S2_arr.shape[1] - np.sum(np.isnan(S2_arr), axis=1)

    # Check if less than 2 replicates
    if np.any(n1 < 2) or np.any(n2 < 2):
        raise ValueError("Some proteins have less than 2 replicates!")
    # Check if n1 and n2 are equal
    if np.any(n1 != n2):
        if equalVar:
            raise ValueError(
                """Number of replicates from both samples
                are not equal, should you use equalVar=False?"""
            )

    # Calculate useful statistics to use in tests
    m1, m2 = np.nanmean(S1_arr, axis=1), np.nanmean(S2_arr, axis=1)
    v1, v2 = np.nanvar(S1_arr, axis=1, ddof=1), np.nanvar(S2_arr, axis=1, ddof=1)

    # Calculate fold-change (Assumes the data is log2 transformed)
    log2fc = m1 - m2

    # Find the index of proteins to be considered for equivalence
    is_test_eq = np.abs(log2fc) < eqThr

    # Calculate the t-test p-values
    ttest_pval = ttest_ind_with_na(
        m1, 
        m2, 
        v1, 
        v2, 
        n1, 
        n2, 
        equal_var=equalVar,
        alternative='two-sided'
    )[1]
    # Apply multiple testing correction
    ttest_pval_corr = multiple_testing_correction(
        ttest_pval,
        correction_type=correction,
        sample_size=None
    )

    # Calculate the tost p-values
    # Test against upper equivalence bound
    p_greater = ttest_ind_with_na(
        m1 + eqThr,
        m2,
        v1,
        v2,
        n1,
        n2,
        equal_var=True,
        alternative="greater"
    )[1]

    # Test against lower equivalence bound
    p_less = ttest_ind_with_na(
        m1 - eqThr,
        m2,
        v1,
        v2,
        n1,
        n2,
        equal_var=True,
        alternative="less"
    )[1]

    # Corect the greater and less individually
    p_greater_corr = multiple_testing_correction(
        p_greater,
        correction_type=correction,
        sample_size=None
    )
    p_less_corr = multiple_testing_correction(
        p_less,
        correction_type=correction,
        sample_size=None
    )
    # Combine the two p-values
    tost_pval = np.maximum(p_greater, p_less)
    tost_pval_corr = np.maximum(p_greater_corr, p_less_corr)

    # Create combination p and q value arrays 
    comb_pval = np.where(
        is_test_eq, 
        tost_pval, 
        ttest_pval
    )
    comb_pval_corr = np.where(
        is_test_eq,
        tost_pval_corr,
        ttest_pval_corr
    )

    # ttest and tost specific significance conditions
    ttest_signf = (ttest_pval_corr < pThr) & (np.abs(log2fc) > dfThr)
    tost_signf = (tost_pval_corr < pThr) & (np.abs(log2fc) < eqThr)

    # Record boolean for significant p values
    is_signf = np.where(
        is_test_eq,
        tost_signf,
        ttest_signf
    )

    # Create test based logging -1 or 1 for p value
    tmp = np.log10(comb_pval)
    logp = np.where(is_test_eq, tmp, -tmp)
    # Create test based logging -1 or 1 for q value
    tmp = np.log10(comb_pval_corr)
    logq = np.where(is_test_eq, tmp, -tmp)

    # Calculate protein status based on significance
    prot_status = np.where(
        is_signf,               # If significant
        np.where(
            is_test_eq,         # If equivalence test
            1.,                 # Eq = 1
            -1.,                # Df = -1
        ),
        0.                      # Not significant = 0
    )

    # Return the results as numpy ndarrays
    return np.stack(
        (
            n1, n2, log2fc, 
            ttest_pval, ttest_pval_corr,
            p_less, p_less_corr,
            p_greater, p_greater_corr,
            tost_pval, tost_pval_corr,
            comb_pval, comb_pval_corr,
            logp, logq, prot_status
        ), 
        axis=1
    )

def ttest_rel_with_na(
        d: np.ndarray,
        n: np.ndarray,
        alternative: str='two-sided'
    ):

    """
        The ttest_rel function from scipy.stats 
        but with the ability to handle missing values.
    """
    # Calculate the degrees of freedom
    df = n - 1
    # Mean difference
    dm = d.mean(axis=1)
    # Calculate the variance of the difference
    v = d.var(axis=1, ddof=1)
    # Calculate the denominator
    denom = np.sqrt(v / n)
    # Calculate t-statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        t = dm / denom
    # Calculate p-value based on alternative hypothesis
    return _ttest_finish(df, t, alternative)

def run_paired(
        S1_arr: np.ndarray,
        S2_arr: np.ndarray,
        pThr: float=0.05,
        dfThr: float=1,
        eqThr: float=0.5, 
        correction: str='fdr', 
    ):
    """
        Run paired t-test on two sets for the 
        given two samples on a paired configuration.
    """    

    # Calculate sample sizes for each protein in each sample
    n1 = S1_arr.shape[1] - np.sum(np.isnan(S1_arr), axis=1)
    n2 = S2_arr.shape[1] - np.sum(np.isnan(S2_arr), axis=1) 
    
    if not np.array_equal(n1, n2):
        raise ValueError("Paired t-test requires n1 and n2 to be the same!")
    # Save the single sample size
    n = n1
    # Calculate the difference between the two samples
    d = (S1_arr - S2_arr).astype('d')
    # logfold change
    log2fc = np.nanmean(d, axis=1)
    
    # Calculate the t-test p-values
    ttest_pval = ttest_rel_with_na(
        d, 
        n, 
        alternative='two-sided'
    )[1]

    # Apply multiple testing correction
    ttest_pval_corr = multiple_testing_correction(
        ttest_pval,
        correction_type=correction,
        sample_size=None
    )

    # Calculate the tost p-values
    # Test against the upper equivalence boundary
    p_greater = ttest_rel_with_na(
        d + eqThr,
        n,
        alternative='greater'
    )[1]
    # Test against the lower equivalence boundary
    p_less = ttest_rel_with_na(
        d - eqThr,
        n,
        alternative='less'
    )[1]
    # Combine the two p-values
    tost_pval = np.maximum(p_greater, p_less)

    # Corect the greater and less individually
    p_greater_corr = multiple_testing_correction(
        p_greater,
        correction_type=correction,
        sample_size=None
    )
    p_less_corr = multiple_testing_correction(
        p_less,
        correction_type=correction,
        sample_size=None
    )

    # Combine the two p-values
    tost_pval = np.maximum(p_greater, p_less)
    tost_pval_corr = np.maximum(p_greater_corr, p_less_corr)
        
    # Check if the difference is within the equivalence boundary
    is_test_eq = np.abs(log2fc) < eqThr

    # Create combination p and q value arrays 
    comb_pval = np.where(
        is_test_eq, 
        tost_pval, 
        ttest_pval
    )
    comb_pval_corr = np.where(
        is_test_eq,
        tost_pval_corr,
        ttest_pval_corr
    )

    # ttest and tost specific significance conditions
    ttest_signf = (ttest_pval_corr < pThr) & (np.abs(log2fc) > dfThr)
    tost_signf = (tost_pval_corr < pThr) & (np.abs(log2fc) < eqThr)

    # Record boolean for significant p values
    is_signf = np.where(
        is_test_eq,
        tost_signf,
        ttest_signf
    )
    
    # Create test based logging -1 or 1 for p value
    tmp = np.log10(comb_pval)
    logp = np.where(is_test_eq, tmp, -tmp)
    # Create test based logging -1 or 1 for q value
    tmp = np.log10(comb_pval_corr)
    logq = np.where(is_test_eq, tmp, -tmp)

    # Calculate protein status based on significance
    prot_status = np.where(
        is_signf,               # If significant
        np.where(
            is_test_eq,         # If equivalence test
            1.,                 # Eq = 1
            -1.,                # Df = -1
        ),
        0.                      # Not significant = 0
    )

    # Return the results as numpy ndarrays
    return np.stack(
        (
            n1, n2, log2fc, 
            ttest_pval, ttest_pval_corr,
            p_less, p_less_corr,
            p_greater, p_greater_corr,
            tost_pval, tost_pval_corr,
            comb_pval, comb_pval_corr,
            logp, logq, prot_status
        ), 
        axis=1
    )

def qEstimate(
        pv, 
        m=None, 
        verbose=False, 
        lowmem=False, 
        pi0=None
    ):
    
    """
    Estimates q-values from p-values
    source: # https://github.com/nfusi/qvalue
    Args
    =====
    m: number of tests. If not specified m = pv.size
    verbose: print verbose messages? (default False)
    lowmem: use memory-efficient in-place algorithm
    pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.
         For most GWAS this is not necessary, since pi0 is extremely likely to be
         1
    Returns
    =====
    qvalues: array of q-values of same size as p-values
    """
    assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

    original_shape = pv.shape
    pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

    if m is None:
        m = float(len(pv))
    else:
        # the user has supplied an m
        m *= 1.0

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(pv > i).sum() for i in np.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        # fit natural cubic spline
        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)
        if verbose:
            print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

        if pi0 > 1:
            if verbose:
                print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            pi0 = 1.0

    assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    if lowmem:
        # low memory version, only uses 1 pv and 1 qv matrices
        qv = np.zeros((len(pv),))
        last_pv = pv.argmax()
        qv[last_pv] = (pi0*pv[last_pv]*m)/float(m)
        pv[last_pv] = -np.inf
        prev_qv = last_pv
        for i in range(int(len(pv))-2, -1, -1):
            cur_max = pv.argmax()
            qv_i = (pi0*m*pv[cur_max]/float(i+1))
            pv[cur_max] = -np.inf
            qv_i1 = prev_qv
            qv[cur_max] = min(qv_i, qv_i1)
            prev_qv = qv[cur_max]

    else:
        p_ordered = np.argsort(pv)
        pv = pv[p_ordered]
        qv = pi0 * m/len(pv) * pv
        qv[-1] = min(qv[-1], 1.0)

        for i in range(len(pv)-2, -1, -1):
            qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

        # reorder qvalues
        qv_temp = qv.copy()
        qv = np.zeros_like(qv)
        qv[p_ordered] = qv_temp

    # reshape qvalues
    qv = qv.reshape(original_shape)

    return qv


# TODO: add more correction methods
def multiple_testing_correction(
        pvalues: np.ndarray, 
        correction_type: str="bonferroni", 
        sample_size: int=None
    ):

    """
    Performs multiple testing correction on p-values 
        using p.adjust methods from R
    Args
    =====
    pvalues: array of p-values
    correction_type: type of correction to perform
    sample_size: number of tests performed, 
        if None, it is set to the length of pvalues
    Returns
    =====
    qvalues: array of q-values of same size as p-values
    """

    if correction_type == None:
        return pvalues
    else:
        pvalues = np.array(pvalues)
        if sample_size is None:
            sample_size = pvalues.shape[0]
        qvalues = np.empty(sample_size)

        if correction_type == "bonferroni":
            qvalues = sample_size * pvalues

        elif correction_type == "holm":
            values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
            values.sort()
            for rank, vals in enumerate(values):
                pvalue, i = vals
                qvalues[i] = (sample_size-rank) * pvalue

        elif correction_type == "fdr":
            by_descend = pvalues.argsort()[::-1]
            by_orig = by_descend.argsort()
            steps = float(len(pvalues)) / np.arange(len(pvalues), 0, -1)
            q = np.minimum(1, np.minimum.accumulate(steps * pvalues[by_descend]))
            qvalues = q[by_orig]
            
        elif correction_type == "qvalue":
            qvalues = qEstimate(pvalues, m=sample_size)

        return qvalues
    
def by_pair(
        cur_comb: tuple[str, str],
        input_path: str,
        res_path: str,
        info_path: str,
        is_log2: bool,
        # cv_filter: bool, TODO: Make without CV filter version
        cv_thr: float=75,  # Can be very large (500) to essentially disable
        p_thr: float=0.05,
        df_thr: float=1,
        eq_thr: float=0.5,
        var_equal: bool=False,
        is_paired: bool=False,
        correction: str='fdr',
        # Adds 90 and 95% CI to the output - A lot slower
        # add_CI: bool=False, TODO: When I solve the time issue, enable CI
    ):
    """
        A main function to bring the sample pairs 
        and test for difference and equivalence 
        on the selected configuration.
    """
    # Logically check the input parameters
    if df_thr < eq_thr:
        raise ValueError(
            """The equivalence boundary must be smaller 
            than the difference boundary (logFC cutoff)!"""
        )

    # Check if the passed correction is valid
    if correction not in ['bonferroni', 'holm','fdr', 'qvalue', None]:
        raise ValueError(
            """Invalid correction method, 
            must be one of 'bonferroni', 'holm','fdr', 'qvalue'!"""
        )
    # Check if the variables passed for variance and paired are logical
    if is_paired and (not var_equal):
        raise ValueError(
            """Paired test cannot be done without equal variance!"""
        )

    S1, S2 = cur_comb
    comb_name = "_vs_".join(cur_comb)
    # Read the data from the hard-drive (feather format)
    S1_data = feather.read_dataframe((input_path+S1+".feather"))
    S2_data = feather.read_dataframe((input_path+S2+".feather"))

    # Check if index of S1 and S2 are the same
    if not (S1_data.index == S2_data.index).all():
        raise ValueError(
            """The two dataframes do not share the same index!"""
        )
    # Get index as proteins
    proteins = S1_data.index

    # Get the data as numpy array
    S1_arr = S1_data.values
    S2_arr = S2_data.values

    # Get the coefficient of variation for each protein
    S1_arr_cv = utils.cv_numpy(S1_arr, axis=1)
    S2_arr_cv = utils.cv_numpy(S2_arr, axis=1)

    # Make the protein selection indicator array
    S1_arr_ps = utils.make_protein_selection_indicator(S1_arr_cv, cv_thr)
    S2_arr_ps = utils.make_protein_selection_indicator(S2_arr_cv, cv_thr)

    # Sum S1 and S2 indicator arrays for selection
    # -2, -1, 0 <- wont be selected
    # 1 <- S1 or S2 will be imputed - selected
    # 2 <- No processing necessary - selected
    S_arr_ps = S1_arr_ps + S2_arr_ps

    # Get the index for 1 (Ms+Qn) and 2s (Qn+Qn)
    subidx = np.where(S_arr_ps >= 2)[0]

    # Subset the dataframes to the shared index
    # and convert them to numpy arrays
    if not is_log2:
        S1_arr = np.log2(S1_arr)
        S2_arr = np.log2(S2_arr)

    # Subset and replace NaN with 1
    S1_arr_ready = S1_arr[subidx]
    # S1_arr_ready[np.isnan(S1_arr_ready)] = 1

    S2_arr_ready = S2_arr[subidx]
    # S2_arr_ready[np.isnan(S2_arr_ready)] = 1

    # Run the tests
    if is_paired:
        # If the test is paired, then run the paired test
        res = run_paired(
            S1_arr_ready,
            S2_arr_ready,
            pThr=p_thr,
            dfThr=df_thr,
            eqThr=eq_thr,
            correction=correction
        )
    else:
        # If the test is unpaired, then run the unpaired test
        res = run_unpaired(
            S1_arr_ready,
            S2_arr_ready,
            pThr=p_thr,
            dfThr=df_thr,
            eqThr=eq_thr,
            equalVar=var_equal,
            correction=correction
        )

    # Create a dataframe from the results
    res_df = pd.DataFrame(
        res,
        columns=COLS,
        index=proteins[subidx]
    )
    # Place S1 and S2 names
    res_df["S1"], res_df["S2"] = S1, S2

    # Write the result dataframe to the hard-drive
    feather.write_dataframe(
        df=res_df, 
        dest=(res_path + comb_name + ".feather") 
    )

    # Creat info data to save
    status_all = np.zeros(len(proteins)) * np.nan
    # status_all[subidx] = res_df["Status"].values
    status_all[subidx] = res[:, -1]

    # Create a info dataframe
    info = pd.DataFrame(
        {
            "Protein": proteins,
            "S1_Status": S1_arr_ps,
            "S2_Status": S2_arr_ps, 
            "Status": status_all, 
            "S1": [S1] * len(proteins),
            "S2": [S2] * len(proteins)
        }
    )
    # Write the info dataframe to the hard-drive
    feather.write_dataframe(
        df=info,
        dest=(info_path + comb_name + ".feather")
    )

def equivalence_percent(
        data: pd.DataFrame, 
        s1_cols: list[str], 
        s2_cols: list[str], 
        pThr: float=0.05, 
        eqThr: float=1.0, 
        correction: str='fdr'
    ):

    """
        Calculates the equivalence percent value between two samples.
    """

    # TODO: Make this function use the same bases as the main testing suite
    # WARNING: At this point this is only used and tested for simulation.

    # Get the values of the two dataframes
    S1_arr = data[s1_cols].values
    S2_arr = data[s2_cols].values

    # Calculate sample sizes for each protein in each sample
    n1 = S1_arr.shape[1] - np.sum(np.isnan(S1_arr), axis=1)
    n2 = S2_arr.shape[1] - np.sum(np.isnan(S2_arr), axis=1)

    # Calculate useful statistics to use in tests
    m1, m2 = np.nanmean(S1_arr, axis=1), np.nanmean(S2_arr, axis=1)
    v1, v2 = np.nanvar(S1_arr, axis=1, ddof=1), np.nanvar(S2_arr, axis=1, ddof=1)
    
    # Calculate the tost p-values
    # Test against upper equivalence bound
    p_greater = ttest_ind_with_na(
        m1 + eqThr,
        m2,
        v1,
        v2,
        n1,
        n2,
        equal_var=True,
        alternative="greater"
    )[1]

    # Test against lower equivalence bound
    p_less = ttest_ind_with_na(
        m1 - eqThr,
        m2,
        v1,
        v2,
        n1,
        n2,
        equal_var=True,
        alternative="less"
    )[1]

    # Calculate the tost p-values
    tost_pval = np.maximum(
        multiple_testing_correction(p_greater, correction_type=correction), 
        multiple_testing_correction(p_less, correction_type=correction)
    )

    # TODO: Calculation of the equivalence percent depends on 
    #  how the total is calculated. 

    return np.sum(tost_pval < pThr) / len(tost_pval)