import re
import time
import feather 

from itertools import combinations, product

import numpy as np # Numerical functions
import pandas as pd # Data handling functions

from sklearn.manifold import TSNE # t-SNE
from sklearn.decomposition import PCA # PCA

from scipy.spatial import distance # Distance Metrics
from scipy.cluster import hierarchy # Hierarchical Clustering

from Bio import SeqIO
from Bio.SeqUtils import molecular_weight

from upsetplot import from_contents

from questvar import test

def select_representative_protein(
        proteins: str
    ):
    """
    Selects a representative protein from a group.

    The function takes a string of protein IDs separated by semicolons. 
    If there's only one ID, it is returned. If there are multiple IDs, 
    the function prioritizes 6-letter IDs over 10-letter ones. If no 6-letter 
    IDs are present, the first ID is returned.

    Parameters:
    proteins (str): A string of protein IDs separated by semicolons.

    Returns:
    str: The ID of the representative protein.
    """
    protein_ids = proteins.split(";")
    
    if len(protein_ids) == 1:
        return protein_ids[0]
    
    six_letter_ids = [id for id in protein_ids if len(id) == 6]
    
    return six_letter_ids[0] if six_letter_ids else protein_ids[0]

def print_series(
        series: pd.Series, 
        header: str = None, 
        tab: int = 0,
        elements_with_order: list = None
    ):
    """
        Print a pandas series with an optional header
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas series")
    if not isinstance(header, str) and header is not None:
        raise TypeError("header must be a string")
    if not isinstance(tab, int):
        raise TypeError("tab must be an integer")
    if not isinstance(elements_with_order, list) and elements_with_order is not None:
        raise TypeError("elements_with_order must be a list")
    if tab < 0:
        raise ValueError(
            """
            tab must be a positive integer amount.Indicating the empty space prior to printing each element
            """
        )

    if header is not None:
        print(header)
    if elements_with_order is not None:
        for i in elements_with_order:
            if i in series.index:
                print(" "*tab, i, "->", series[i])
    else:
        for index, value in series.items():
                print(" "*tab, index, "->", value)

# Timer Related Functions
def getTime():
    """Get the current time for timer"""
    return time.time()

def prettyTimer(seconds):
    """Better way to show elapsed time"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02dh:%02dm:%02ds" % (h, m, s)

def getMW(seq):
    try: 
        return molecular_weight(seq, "protein") / 1000
    except ValueError:
        # error thrown, for example, when dealing w/ X amino acid
        return np.nan
    
def remove_sequences_with_invalid_AA(seq):
    # Invalid characters are: X, B, Z, J, U, O, *
    if re.search("[X|B|Z|J|U|O|\*]", seq):
        return np.nan
    else:
        return seq

def check_length(peptide):
    return len(peptide) >= 7 and len(peptide) <= 50

def parse_proteome_header(header):
    """
    Parse a Uniprot fasta proteome header and return a dictionary of informative variables.

    Args:
        header (str): Uniprot fasta proteome header string.

    Returns:
        dict: A dictionary of informative variables.
    """
    # Define regular expressions for each field we want to extract
    regexes = {
        "reviewStatus": r"^(sp|tr)\|([A-Z0-9]+)\|",
        "entry": r"^.*?\|([A-Z0-9-]+)\|",
        "entryName": r"^.*?\|([A-Z0-9]+)_([A-Z]+)",
        "geneName": r" GN=([^ ]+)",
        "proteinDescription": r" ([^=]+)(?<! [A-Z]{2}=)",
    }

    # Extract the information using regular expressions
    variables = {}
    for key, regex in regexes.items():
        match = re.search(regex, header)
        if match:
            if key == "entryName":
                variables[key] = f"{match.group(1)}_{match.group(2)}" if match.group(2) else match.group(1)
            else:
                variables[key] = match.group(1) if key == "proteinDescription" else match.group(1)
                if key == "proteinDescription":
                    variables[key] = variables[key].strip(" OS")

    return variables

# Function to read and parse the fasta file into a dataframe
def fasta_to_df(reference_path, fasta_ID):
    """
    Read and parse the Uniprot fasta proteome file into a dataframe.

    Args:
        reference_path (str): Path to the Uniprot fasta proteome file.
        fasta_ID (str): The ID of the fasta file.

    Returns:
        pandas.DataFrame: A dataframe containing the information from the fasta file.
    """
    # Initialize a list to hold all the entries
    results = []
    for record in SeqIO.parse(reference_path, "fasta"):
        # Remove sequences with invalid amino acids
        cur_seq = remove_sequences_with_invalid_AA(str(record.seq))
        if cur_seq is np.nan: continue # Skip sequences if with invalid amino acids
        if len(cur_seq) <= 7: continue # Skip sequences with length less than 7
        # Parse the header
        cur_dict = parse_proteome_header(record.description)
        # Skip sequence if gene name is not available 
        if "geneName" not in cur_dict: continue
        # Get the sequence
        cur_dict["sequence"] = cur_seq
        # Add the sequence length
        cur_dict["sequenceLength"] = len(cur_seq)
        # Add molecular weight
        cur_dict["molecularWeight_kDa"] = getMW(cur_seq)
        # Add the entry to the list
        results.append(cur_dict)

    # Convert the list to a dataframe
    df = pd.DataFrame(results)
    # Add the fasta ID
    df["fastaId"] = fasta_ID
    # Update the review status
    df["reviewStatus"] = df["reviewStatus"].apply(lambda x: "reviewed" if x == "sp" else "unreviewed") 
    # Update the isoform status
    # If the entry name contains a dash, it is an isoform
    df["isoformStatus"] = df["entry"].apply(lambda x: "isoform" if "-" in x else "canonical")
    # This is not perfect, but it is the best we can do with the information we have 
    # Order the columns
    df = df[[
        "fastaId", "entry", "entryName", "geneName", "proteinDescription",
        "reviewStatus", "isoformStatus", "sequenceLength", "molecularWeight_kDa", "sequence"
    ]].sort_values(by=["entry", "isoformStatus"], ascending=[True, False])

    # Return the dataframe
    return df.reset_index(drop=True)

def scale_the_data(
        data: pd.DataFrame, 
        method: str="zscore",
        axis: int=1, 
        is_log: bool=False
    ):
    """
    Scale the data using specified method on the specified axis.
    """

    # Check the data type
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "The data type is not supported, please use either a pandas DataFrame"
        )
    
    # Check the axis
    if axis not in [0, 1]:
        raise ValueError(
            "The axis should be either 0 (column-wise) or 1 (row-wise)."
        )

    # Check the method
    if method not in ["zscore", "minmax", "foldchange", "log2", "log10"]:
        raise ValueError(
            "The method should be either zscore, minmax, foldchange, log2, or log10."
        )

    idx = data.index
    cols = data.columns

    # Z-score Standardization
    if method == "zscore":
        if axis == 0:
            res = (
                data.values - 
                data.mean(axis=axis).values
            ) / data.std(axis=axis).values
        else:
            res = (
                data.values - 
                data.mean(axis=axis).values.reshape(-1, 1)
            ) / data.std(axis=axis).values.reshape(-1, 1)
    # Min-Max scaling
    elif method == "minmax":
        if axis == 0:
            res = (
                data.values - data.min(axis=axis).values
            ) / (
                data.max(axis=axis).values - 
                data.min(axis=axis).values
            )
        else:
            res = (
                data.values - 
                data.min(axis=axis).values.reshape(-1, 1)
            ) / (
                data.max(axis=axis).values.reshape(-1, 1) - 
                data.min(axis=axis).values.reshape(-1, 1)
            )
    # Fold-change scaling
    elif method == "foldchange":
        if axis == 0:
            if is_log:
                res = (
                    data.values - data.mean(axis=axis).values
                )
            else:
                res = (
                    data.values / data.mean(axis=axis).values
                )
        else:
            if is_log:
                res = (
                    data.values - 
                    data.mean(axis=axis).values.reshape(-1, 1)
                )
            else:
                res = (
                    data.values / 
                    data.mean(axis=axis).values.reshape(-1, 1)
                )
    elif method == "log2":
        res = np.log2(data.values)
    elif method == "log10":
        res = np.log10(data.values)

    
    return pd.DataFrame(
        res, 
        index=idx, 
        columns=cols
    )

def cv_numpy(
        x: np.ndarray, 
        axis: int = 1, 
        ddof: int = 1, 
        ignore_nan: bool = False, 
        format: str = "percent"
    ):
    """
        Calculates the coefficient of variation of the values in the passed array.
        Parameters
        ----------
        x : array_like
            Input array or object that can be converted to an array.
        axis : int or None, optional
            Axis along which the coefficient of variation is computed. Default is 1.
            If None, compute over the whole array `x`.
        ddof : int, optional
            Means Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            By default `ddof` is one.
        ignore_nan : bool, optional
            A flag indicating whether to propagate NaNs. If True, NaNs will be
            ignored. If False, NaNs will be propagated, and NaN outputs will
            result.
        format : str, optional
            The format of the output. Default is "ratio".
            "ratio" : Returns the coefficient of variation as a ratio.
            "percent" : Returns the coefficient of variation as a percentage.
        Returns
        -------
        coefficient_of_variation : ndarray
            The coefficient of variation, i.e. standard deviation divided by the
            mean.
    """
    # Check if x is a numpy array
    if not isinstance(x, np.ndarray):
        try: 
            x = np.asarray(x)
        except:
            raise TypeError("Input x must be an array-like object.")
        
    # Check if axis is an integer
    if not isinstance(axis, int):
        raise TypeError("Input axis must be an integer. [0,1]")
    
    # Check if ddof is an integer
    if not isinstance(ddof, int):
        raise TypeError("Input ddof must be an integer.")

    # If ignore_nan use np.nanstd and np.nanmean
    if ignore_nan:
        cv = np.nanstd(x, axis=axis, ddof=ddof) / np.nanmean(x, axis=axis)
    else:
        cv = np.std(x, axis=axis, ddof=ddof) / np.mean(x, axis=axis)

    if format == "ratio":
        return cv
    elif format == "percent":
        return cv * 100

def make_protein_selection_indicator(
      cv_arr: np.ndarray,
      cv_thr: float = 0.15
    ):
    """
        Make the protein selection indicator array
        -1: CV > cv_thr (Unreliably Quantified)
        0: NaN (Not Quantified)
        1: CV <= cv_thr (Robustly Quantified)

        Parameters
        ----------
        cv_arr: np.ndarray
            Array of CV values
        cv_thr: float
            CV threshold

        Returns
        -------
        arr: np.ndarray
            Array of protein selection indicator            
    """

    # invalid cv values
    cv_arr[cv_arr < 0] = np.nan

    # find index for NaN
    nan_idx = np.where(
        np.isnan(cv_arr)
    )[0]
    # find index for CV > cv_thr
    cv_idx = np.where(cv_arr > cv_thr)[0]
    # find index for others
    comp_idx = np.where(
        (cv_arr <= cv_thr) & (~np.isnan(cv_arr))
    )[0]

    # Build the array
    arr = np.zeros(len(cv_arr))
    arr[cv_idx] = -1
    arr[nan_idx] = 0
    arr[comp_idx] = 1

    # Return the array
    return arr

def pick_best_replicates(
        subset: pd.DataFrame, 
        N: int = 3,
        verbose: bool = False
    ):
    """
        A method to picks the best N replicates based on 
        the median CV of all proteins in a given combination.
    """
    cv_weight = 0.25
    cp_weight = 0.75
    cols = subset.columns
    # Get all the combinations
    combs = list(combinations(cols, N))
    # Initialize the best combination
    best_comb = combs[0]
    best_score = 0
    # Loop through each combination
    for n , cur_cols in enumerate(combs):
        cur_subset = subset[list(cur_cols)]
        cur_cv = np.nanmedian(cv_numpy(cur_subset))
        cur_cp = ((~(cur_subset.isna())).sum(axis=1) == len(cur_cols)).sum() / len(cur_subset)
        cur_score = (cv_weight * (1 - cur_cv)) + (cp_weight * cur_cp)
        if verbose:
            print(
                f"Combination {n+1} of {len(combs)}: {cur_cols} -> CV: {cur_cv:.2f}, Completeness: {cur_cp:.2f}, Score: {cur_score:.2f}"
            )
        
        if cur_score > best_score:
            best_score = cur_score
            best_comb = cur_cols
    
    if verbose:
        print(f"Best Combination: {best_comb} -> Score: {best_score:.2f}")

    return list(best_comb)

def create_pair_groupings(
        combs_lst: list[tuple[str, str]],
        metadata: pd.DataFrame, 
        id_col: str = "Cell_line", 
        pair_str: list[str] = ["S1", "S2"]
    ):
    """
        Function to create a grouping information to represent 
        pairs S1 and S2 metadata in a wide format.
    """
    # Check if the combs_lst is valid
    if not isinstance(combs_lst, list):
        raise TypeError("combs_lst must be a list of pairs")
    # Check if the metadata is valid
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas dataframe")
    # Check if the pair_str is valid
    if not isinstance(pair_str, list):
        raise TypeError(
            "pair_str must be a list of two strings representing S1 and S2 column IDs"
        )
    
    # Check if id_col is in metadata or is the index
    if id_col not in metadata.columns and id_col != metadata.index.name:
        raise ValueError("id_col is not in metadata")
    elif id_col == metadata.index.name:
        # Remove the index to make it a column
        metadata = metadata.reset_index()    

    # Make a numpy array of the combs_lst
    combs_arr = np.array(combs_lst)
    # Metadata for S1
    tmp1 = metadata.set_index(id_col).loc[combs_arr[:, 0]].reset_index()
    tmp1.columns = [(i+"_"+pair_str[0]) for i in tmp1.columns]
    # Metadata for S2
    tmp2 = metadata.set_index(id_col).loc[combs_arr[:, 1]].reset_index()
    tmp2.columns = [(i+"_"+pair_str[-1]) for i in tmp2.columns]

    # Combine the two metadata
    groupings = pd.concat([tmp1, tmp2], axis=1).rename(columns={
        (id_col+"_"+pair_str[0]): pair_str[0],
        (id_col+"_"+pair_str[-1]): pair_str[-1]
    })
    # Return the groupings
    return groupings


def create_pair_metadata_labels(
        groupings: pd.DataFrame, 
        var_lst: list, 
        pair_str: list[str] = ["S1", "S2"],
        add_ordered_ver: bool = True,
        add_compare_ver: bool = True,
        new_var_lst: list = None,
        sep: str = " & "
    ):
    """
        Function to get S1 and S2 labels for variables passed
    """
    
    # get groupings columns without _S1 and _S2
    org_cols = [i.split("_")[0] for i in groupings.columns]

    # Check if var_lst is a list
    if not isinstance(var_lst, list):
        raise TypeError("var_lst must be a list of strings")
    elif len(var_lst) == 0:
        raise ValueError("var_lst must not be empty")

    # Check if pair_str is a list
    if not isinstance(pair_str, list):
        raise TypeError("pair_str must be a list of two strings")

    # Check if new_var_lst is a list
    if new_var_lst is not None:
        if not isinstance(new_var_lst, list):
            raise TypeError("new_var_lst must be a list of strings")
        elif len(new_var_lst) != len(var_lst):
            raise ValueError("new_var_lst must be the same length as var_lst")

    # if both add_ordered_ver and add_compare_ver are False
    if not add_ordered_ver and not add_compare_ver:
        raise ValueError("At least one of add_ordered_ver and add_compare_ver should be True")

    # Initialize a new dataframe to store labels
    new_df = groupings[pair_str].copy()

    # Loop through the variables
    for i in var_lst:
        # Create columns with variable and pair
        cols = [i+"_"+j for j in pair_str]
        # Initialize the new column
        new_col = i
        if new_var_lst is not None:
            new_col = new_var_lst[var_lst.index(i)]
        # Create columns to save ordered and compare versions
        compare_col = new_col + "_Compare"
        ordered_col = new_col # + "_Ordered"

        # Get the tuple of the values
        grp = groupings[cols].apply(
            lambda x: x.unique().tolist(),
            axis=1
        )
        if add_compare_ver:
            new_df[compare_col] = grp.map(
                lambda x: sep.join(x),
            )

        if add_ordered_ver:
            new_df[ordered_col] = grp.map(
                lambda x: sep.join(sorted(x)),
            )
    

    return new_df

def collect_status_for_protein_matrix(
        info_file: str,
    ):
    """
        Collects the status from info data 
        to form protein matrix later on.
    """

    # Read the data from the info files
    a = feather.read_dataframe(info_file)
    # Create a key - value return key: (S1, S2), value: status
    return (
        (
            a.loc[0, "S1"], 
            a.loc[0, "S2"]
        ),
        a["Status"].tolist()
    )

def apply_dimensional_reduction(
        data: pd.DataFrame,
        method: str,
        metadata: pd.DataFrame,
        metaindex: str,
        metacols: list,
        n_components: int=2,
        **kwargs 
    ) -> pd.DataFrame:
    """
        Apply a dimensional reduction method to the data.
    """

    # If the method is PCA
    if method == "pca":
        # Initialize the model
        model = PCA(
            n_components=n_components, 
            **kwargs
        )
    # If the method is t-SNE
    elif method == "tsne":
        # Initialize the model
        model = TSNE(
            n_components=n_components, 
            **kwargs
        )
    else:
        raise ValueError(
            "The passed method is not supported!",
            "Please use one of the following: pca, tsne"
        )
    
    # Fit the model 
    res = model.fit_transform(data)

    # Create a dataframe with the results   
    res = pd.DataFrame(
        res,
        index=data.index,
        columns=["Dim_{}".format(i) for i in range(1, n_components+1)]
    ).join(
        metadata.set_index(metaindex)[metacols]
    )

    return model, res

def make_upset_data(
    data: pd.DataFrame,
    stacked: bool = False,
    main_id_col: str = "query",
    feature_col: str = "native",
    secondary_id_col: str = "source", # TODO: Should allow multiple columns in Future
    concat_str: str = " | ",
    ):
    """
    """
    # Check if main_id_col is in the data
    if main_id_col not in data.columns:
        raise ValueError(
            f"""Column '{main_id_col}' not in the data."""
        )
    # Check if feature_col is in the data
    if feature_col not in data.columns:
        raise ValueError(
            f"""Column '{feature_col}' not in the data."""
        )
    if stacked:
        # concat_str must be a string
        if not isinstance(concat_str, str):
            raise TypeError(
                "concat_str must be a string"
            )
        # secondary_id_col must be in the data
        if secondary_id_col not in data.columns:
            raise ValueError(
                f"""Column '{secondary_id_col}' not in the data."""
            )
        # Create a unique id with the source and query
        # TODO: Make this more general for multiple secondary_id_cols
        # WARNING: This works with a single element in secondary_id_cols
        data["id"] = data[secondary_id_col] + concat_str + data[feature_col]
        feat_col = "id"
    else:
        feat_col = feature_col

    # Dictionary to store the terms in query
    terms_dict = {}
    for q in data[main_id_col].unique():
        terms_dict[q] = data[data[main_id_col] == q][feat_col].unique().tolist()

    # Create a upset plot data from dictionary of terms
    upset_data = from_contents(terms_dict)

    if stacked:
        # Add source and term columns with split id
        # TODO: Make this more general for multiple secondary_id_cols
        # WARNING: This works with a single element in secondary_id_cols
        upset_data[secondary_id_col] = upset_data["id"].str.split(concat_str).str[0]
        upset_data[feature_col] = upset_data["id"].str.split(concat_str).str[-1]
        # Drop the id column
        upset_data.drop("id", axis=1, inplace=True)
    
    # Return the upset data
    return upset_data

def custom_clustering(
        data: pd.DataFrame,
        # TODO: Add Different Clustering Methods
        clustering: str = "hierarchical", 
        nclusters: int = 6,
        method: str = "ward",
        metric: str = "euclidean",        
    ):
    """
        Applys custom clustering to the data with scipy.cluster.hierarchy
        returns clusters and linkage matrix.
    """
    if clustering == "hierarchical":
        # Calculate the distance matrix
        dist_mat = distance.pdist(data, metric=metric)
        # Calculate the linkage matrix
        linkage_mat = hierarchy.linkage(dist_mat, method=method)
        # Calculate the clusters
        clusters = hierarchy.fcluster(linkage_mat, nclusters, criterion="maxclust")
    else:
        raise ValueError("Clustering methods other than hierarchical not implemented.")

    # Convert the clusters to a pandas series
    clusters = pd.Series(clusters, index=data.index, name="cluster")

    # Return the clusters and linkage matrix
    return clusters, linkage_mat

def get_term_from_data(
        gp_res: pd.DataFrame, 
        spec_term: str,
        queries: list[str] = None,
        queries_col: str = "query",
        term_col: str = "native",
        term_name_col: str = "name",
        
    ):
    """
    Validate and retrieve data for the custom scatterplot.

    Parameters:
    - gp_res: DataFrame containing the gene pathway enrichment results.
    - metadata: DataFrame containing metadata information.
    - spec_term: Specific term for analysis.

    Returns:
    - sub_gp_res: DataFrame containing the subset of data for plotting.
    - term_name: Name of the specific term.
    """
    try:
        # Check if spec_term is present in gp_res
        if spec_term not in gp_res[term_col].unique():
            raise ValueError(f"Term {spec_term} not found in gp_res.")

        # Get the term name
        term_name = gp_res[gp_res[term_col] == spec_term][term_name_col].unique()[0]

        # Filter and process data
        sub_gp_res = gp_res[gp_res[term_col] == spec_term].reset_index(drop=True)

        # Check if queries are present in the data
        if queries is not None:
            if not isinstance(queries, list):
                raise TypeError("queries must be a list.")
            # If queries are passed and valid
            else:
                # Add missing queries to the data
                ts2add = set(queries) - set(sub_gp_res[queries_col])
                sub_gp_res = sub_gp_res.append(
                    pd.DataFrame({queries_col: list(ts2add)})
                ).set_index(queries_col).loc[queries].reset_index()

        return sub_gp_res, term_name

    except Exception as e:
        print(f"Error: {e}")
        return None, None

def pick_terms(
        data: pd.DataFrame,
        pivot_data: pd.DataFrame,
        source: str,
        source_col: str = "source",
        full_data: pd.DataFrame = None,
        terms: list[str] = None,
        non_significant: bool = False,
        pick_method: str = "cluster", # 'top', 'cluster', 'shared', 'group', 'random', 'single'
        pick_number: int = 5,
        sort_by: str = None,
        sort_col: str = None,
        name_col: str = "name",
        queries: list[str] = None,
        ascending: bool = False,
        verbose: bool = True
    ):
    """
        Function that allows picking terms from 
        the enrichment analysis results with custom ordering.
    """
    start = getTime()

    if non_significant:
        if full_data is None:
            raise ValueError("If non_significant is True, full_data should be provided.")
        if isinstance(full_data, pd.DataFrame):
            if full_data.shape[0] == 0:
                raise ValueError("full_data is empty.")
        else:
            raise TypeError("full_data should be a pandas DataFrame.")
        if verbose:
            print("Including non significant terms in other groups as complementary.")
    else:
        if verbose:
            print("Excluding non significant terms from other groups")
    
    if queries is None:
        queries = pivot_data.columns.tolist()

    if sort_by is not None:
        if ascending is None:
            ascending = False
        # Check if the sort_by is in the pivot_data columns
        if sort_by == "column":
            verb_term = "by single group column"
            if sort_col is None:
                raise ValueError("If sort_by is column, sort_col should be provided.")
            if sort_col not in pivot_data.columns:
                raise ValueError(f"{sort_col} is not in the pivot_data columns.")
            
            # Sort the pivot data by the provided group column
            pivot_data = pivot_data.sort_values(
                sort_col,
                ascending=ascending
            )
        else:
            if sort_by not in ["mean", "median", "min", "max", "sum"]:
                raise ValueError("if sort_by not column it should be from: [mean, median, min, max, sum]")
            else:
                verb_term = f"by terms' {sort_by}"

            # Sort the pivot data by the stats of the terms of all groups
            pivot_data = pivot_data.loc[
                pivot_data.apply(
                    sort_by,
                    axis=1
                ).sort_values(
                    0,
                    ascending=ascending
                ).index.tolist()
            ]

        if ascending:
            if verbose:
                print(f"Sorting the terms {verb_term} in ascending order.")
        else:
            if verbose:
                print(f"Sorting the terms {verb_term} in descending order.")

    if terms is not None:
        # Check if terms is a list
        if not isinstance(terms, list):
            raise TypeError("terms should be a list of strings.")
        pick_number = len(terms)
        if pick_number < 2:
            raise ValueError("At least 2 terms should be passed to use for plotting.")
        if verbose:
            print(f"Selecting {pick_number} terms passed by user.")
    else:
        if pick_method is None:
            raise ValueError("Either terms or pick_method should be provided.")
        
        if pick_number is None:
            raise ValueError("If pick_method is provided, pick_number should be provided as well.")
        if pick_number < 2:
            raise ValueError("pick_number should be greater than 0.")
        
    if pick_method == "top":
        # Select the top n terms based on the number of missing values
        terms2plot = pivot_data.sum(
            axis=1
        ).sort_values(
            ascending=ascending
        ).index.tolist()[:pick_number]
    
    elif pick_method == "random":
        # Select n random terms
        terms2plot = pivot_data.sample(
            n=pick_number
        ).index.tolist()
        
    elif pick_method == "single":
        if sort_by is None:
            raise ValueError("sort_group should be specified for pick_method single")
        # Pick top n terms from single group
        terms2plot = pivot_data[pivot_data[sort_by]].sum(
            axis=1
        ).sort_values(
            ascending=ascending
        ).index.tolist()[:pick_number]

    elif pick_method == "cluster":
        # Select n terms from each cluster
        # Add cluster information to the pivot table
        terms2plot = pivot_data.join(
            data.set_index(name_col)[[pick_method]]
        ).reset_index().drop_duplicates().sort_values(
            [pick_method, name_col],
            ascending=[True, False]
        ).groupby(
            pick_method
        ).head(
            pick_number
        ).reset_index(
            drop=True
        )[name_col].tolist()

    elif pick_method == "shared":
        terms2plot = []
        for i in range(1, len(queries)+1):
            terms2plot.extend(
                pivot_data[
                    (~(pivot_data.replace(0, np.nan).isna())).sum(axis=1) == i
                ].head(pick_number).index.tolist()
            )

    elif pick_method == "group":
        tmp_data = pivot_data.copy()
        tmp_data = ~(tmp_data.replace(0, np.nan).isna())
        terms2plot = []
        for i in range(1, len(queries)+1):
            cur_comb = list(combinations(queries, i))
            for comb in cur_comb:
                tmp = (tmp_data.loc[:, comb])
                tmp = (
                    (tmp.all(axis=1)) & 
                    (tmp_data.sum(axis=1) == len(comb))
                )
                terms2plot.extend(tmp[tmp==True].index.tolist()[:pick_number])
    else:
        raise ValueError("pick_method not supported should be one of: [top, cluster, shared, group, random, single]")

    if verbose:
        print(f"Selecting {pick_number} terms using {pick_method} method.")

    
    if non_significant:
        return full_data[
            (full_data[name_col].isin(terms2plot)) & 
            (full_data[source_col] == source) 
        ]
    
    else:    
        return data[
            data[name_col].isin(terms2plot) &
            (data[source_col] == source)
        ]


def mean_pairwise_corr(
        data: pd.DataFrame, 
        s1_cols: list[str], 
        s2_cols: list[str], 
        method: str="pearson"
    ):
    """
        Calculates the mean pairwise correlation between
        two distinct samples with their replicates
    """
    corr_mat = data.corr(method=method)
    s1_indices = [corr_mat.columns.get_loc(c) for c in s1_cols]
    s2_indices = [corr_mat.columns.get_loc(c) for c in s2_cols]
    i, j = np.meshgrid(s1_indices, s2_indices, indexing='ij')
    return corr_mat.values[i.ravel(), j.ravel()].mean()

def collect_metrics(
        data: pd.DataFrame,
        s1_cols: list[str], 
        s2_cols: list[str], 
        pThr: float=0.05, 
        eqThr: float=1.0, 
    ):
    """
        Collection function to run correlation and 
        percent equivalence for the given conditions.
    """
    return {
        "pearson (r)": mean_pairwise_corr(
            data, 
            s1_cols, 
            s2_cols, 
            method="pearson"
        ),
        "spearman (p)": mean_pairwise_corr(
            data, 
            s1_cols, 
            s2_cols, 
            method="spearman"
        ),
        "kendall (tau)": mean_pairwise_corr(
            data, 
            s1_cols, 
            s2_cols, 
            method="kendall"
        ),
        "SEI": test.equivalence_percent(
            data,
            s1_cols,
            s2_cols,
            pThr=pThr,
            eqThr=eqThr
        )
    }

def save_result_as_frame(
        res_dict: dict,
        identifier: str,
        total_fts: int,
        use_prctg: bool = True
    ):
    """
        Function that converts a result dictionary 
        from simulation loops into a dataframe
    """

    data = pd.DataFrame.from_dict(
        res_dict
    ).T.reset_index().rename(
        columns={"index": "n_fts"}
    ).melt(
        id_vars="n_fts",
        var_name="metric",
        value_name="value"
    )

    data["identifier"] = identifier

    if use_prctg:
        # Convert Protein Number to Percentage of Changed Proteins
        data["n_fts"] = data["n_fts"].astype(float)
        data["n_fts"] = data["n_fts"] / total_fts * 100

    return data
