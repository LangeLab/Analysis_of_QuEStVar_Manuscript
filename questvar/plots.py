import pandas as pd # Data Handling
import numpy as np # Numerical python array

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec # For complex figure layouts
from matplotlib import patches as mpatches
from upsetplot import UpSet

from itertools import combinations

import warnings; warnings.filterwarnings('ignore')

from questvar import utils


def_colors = [
    '#2274A5', '#FFBF00', "#999999", 
    '#0B3C49', '#32936F', '#FC6471'
]
stat_colors = [
    "#29335c", "#7e8cc6", "#565d61",
    "#ff8020", "#eeeeee", "#70161e",
    "#d06770"
]
enrichment_palette = {
    "GO:BP": "#5a189a",
    "GO:CC": "#7b2cbf",
    "GO:MF": "#9d4edd",
    "KEGG":  "#f48c06",
    "REAC":  "#ffba08"
}

def save_figures(
        fig_obj,
        filename: str,
        filepath: str = '',
        fileformat: list[str] = ['png', 'svg', 'pdf'],
        dpi: int = 300,
        transparent: bool = True
    ):
    """
        Function to save a given plot object in given list of formats.
    """
    for i in fileformat:
        fig_obj.savefig(
            filepath + "/" + i + "/" + filename + '.' + i,
            format=i,
            dpi=dpi, 
            transparent=transparent,
            bbox_inches='tight',
            pad_inches=0.01
        )

def hex_to_rgb(
        hex_code: str,
    ):
    """
        Converts a hex color code to RGB
    """
    hex_code = hex_code.lstrip('#').rstrip(";")
    lv = len(hex_code)
    return tuple(
        int(hex_code[i:i + lv // 3], 16) 
        for i in range(0, lv, lv // 3)
    )

def pick_color_based_on_background(
        bgColor: str,
        lightColor: str = "#FFFFFF",
        darkColor: str = "#000000",
        hex: bool = False,
        rgb: bool = False,
        uicolor: bool = False,
    ):
    """
        Picks a light or dark color based on the background color
        Built from and answer on StackOverflow:
            https://stackoverflow.com/a/76275241
    """
    pass
    if hex:
        color = bgColor.lstrip("#").rstrip(";")
        r, g, b = hex_to_rgb(color)
        uicolors = [r/255, g/255, b/255]
    elif rgb:
        r, g, b = bgColor
        uicolors = [r/255, g/255, b/255]
    elif uicolor:
        uicolors = bgColor
    else:
        raise ValueError(
            """Please turn on one of the color modes relevant to bgColor passed 
            Options: hex, rgb, uicolor."""
        )

    adjusted = []
    for col in uicolors:
        col2 = col
        if col <= 0.03928:
            col2 = col/12.92
        
        col2 = ((col2+0.055)/1.055)**2.4
        adjusted.append(col2)

    L = 0.2126 * adjusted[0] + 0.7152 * adjusted[1] + 0.0722 * adjusted[2]

    return darkColor if L > 0.179 else lightColor

def color_palette(
        pal: list, 
        size: int = 1, 
        name: str = "default colors",
        save: bool = False,
        filename: str = 'default_colors_pal',
        fileformat: str = 'png',
        filepath: str = '',
        vectorized: bool = False,
        dpi: int = 100
    ):    

    """
        Customized sns.palplot allowing either showing or saving 
            the palette as a figure.
        if a dictionary is passed, the keys will be used as labels 
            and written inside the boxes of the palette.
    """
    # Check if the palette is a dictionary
    if isinstance(pal, dict):
        # Get the labels
        labels = list(pal.keys())
        # Get the colors
        pal = list(pal.values())
    else:
        # Check if the palette is a list
        if not isinstance(pal, list):
            raise ValueError(
                "The palette should be either a list or a dictionary"
            )
        # Get the labels
        labels = ["" for _ in range(len(pal))]

    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    # Set the labels as text inside the boxes diagonally
    for i, label in enumerate(labels):
        ax.text(
            i, 0, label, 
            rotation=45, 
            rotation_mode="anchor",
            ha="center", 
            va="center", 
            # Color is picked based on the background color
            color=pick_color_based_on_background(
                pal[i],
                hex=True,
            ),
            fontsize=11,
        )
    # Ensure nice border between colors
    ax.set_xticklabels(["" for _ in range(n)])
    # The proper way to set no ticks
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(
        f"Palette for {name}", 
        fontsize=14,
        loc="left"
    )
    # Remove all axes
    ax.axis('off')
    
    if save:
        if vectorized:
            if fileformat in ['png', 'jpg', 'jpeg']:
                raise ValueError(
                    'Vectorized images can only be saved as svg, pdf or eps'
                )
            transparent = True
            

        f.savefig(
                filepath + filename + '.' + fileformat,
                format=fileformat,
                dpi=dpi, 
                transparent=transparent,
                bbox_inches='tight',
                pad_inches=0
            )
    else:
        plt.show()

def check_cv(
        data: pd.DataFrame,
        loop_dict: dict,
        hue_dict: dict,
        hue_name: str = "Tissue_type",
        picked_quantile: float = 0.95,
        title_add: str = "",
        point_color: str = def_colors[3],
        line_color: str = def_colors[5],
    ):
    """
    """

    # picked quantile check
    if picked_quantile < 0 or picked_quantile > 1:
        raise ValueError("picked_quantile should be between 0 and 1")

    # Initialize the CV data to store the results
    cv_data = pd.DataFrame()
    # Loop over the samples
    for group, samples in loop_dict.items():
        # Calculate the CV per sample
        if len(samples) > 1: 
            cv_data[group] = np.abs(
                utils.cv_numpy(
                    data[samples].values,
                    format="percent",
                    ignore_nan=True,
                )
            )

    # Calculate single sample CV based on the picked quantile
    if picked_quantile == 0.5:
        cv_ser = cv_data.median()
        method_name = "Median"
    else:
        cv_ser = cv_data.quantile(picked_quantile)
        method_name = f"{picked_quantile}% Quantile"

    plot_data = pd.DataFrame(
        {
            "CV": cv_ser,
            hue_name: pd.Series(hue_dict),
            "Protein": (~cv_data.isna()).sum()
        }
    ).sort_values("Protein")


    # Create a plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=plot_data, y=hue_name, x="CV", color="white")
    sns.stripplot(data=plot_data, y=hue_name, x="CV", color=point_color, alpha=0.5)
    plt.axvline(x=plot_data["CV"].median(), color=line_color, linestyle="--", label="Median", linewidth=2)
    plt.xlabel(f"{method_name} CV")
    plt.ylabel("")
    plt.title(f"CV per Samples Grouped by {hue_name} \n {title_add}")
    sns.despine(left=True)

    return plt

def check_outliers():
    pass

def check_normality():
    pass

def check_correlation():
    pass

def check_distribution(
        data: pd.DataFrame,
        loop_dict: dict, 
        hue_dict: dict,
        color_dict: dict, 
        group_name: str = "Cell_line",
        hue_name: str = "Tissue_type",
        feature_name: str = "Protein",
        title_add: str = "",
    ):
    """
    """
    # Kdeplot of all samples colored by tissue
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop through all samples
    for group, samples in loop_dict.items():
        # Find protein means for given sample
        val = data[samples].mean(axis=1).values
        val = val[~np.isnan(val)]

        # Plot the distribution of the sample
        sns.kdeplot(
            val,
            color=color_dict[hue_dict[group]],
            linewidth=1,
            ax=ax,
            fill=False,
        )
    # Set the x-axis label
    ax.set_xlabel(feature_name+" Abundance")
    # Set the y-axis label
    ax.set_ylabel("Density")
    # Set the title
    ax.set_title(
        feature_name + 
        " Abundance Distributions of Averaged " + 
        group_name + 
        " - " + 
        title_add
    )

    # Create a legend
    handles = [
        plt.Line2D(
            [], [],
            color=cl,
            marker="o",
            linestyle="",
            label=ts,
        ) for ts, cl in color_dict.items()
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        title=hue_name,
        prop={"size": 10},
    )

    sns.despine()
    plt.tight_layout()
    
    # Return the plot
    return fig

def check_completeness(
        data: pd.DataFrame, 
        loop_dict: dict, 
        hue_dict: dict,
        color_dict: dict, 
        complete_threshold: float = 0.6, # Should be between 0 and 1
        group_name: str = "Cell_line",
        hue_name: str = "Tissue_type",
        feature_name: str = "Proteins",
        title_add: str = "",
        label_color: str = "black",
        label_x_offset: int = -25,
        label_y_offset: int = 200,
    ):
    """
    Function to check the data completeness by looking 
    at the features based on the defined groups and conditions

    Parameters
    ----------
    data: pd.DataFrame
        The quantitative data to be used in the analysis
    loop_dict: dict
        A dictionary of the grouping features and a list of 
        samples names in each group 
        (eg. cell line sample - its replicates)
    hue_dict: dict
        A dictionary of the grouping features and a 
        coloring variable for the grouping feature 
        (eg. cell line sample - Tissue type)
    color_pal: dict
        A dictionary of the grouping features and a 
        color to be used in the plots
    """

    ## Checks
    # If loop_dict's keys and hue_dict's keys are not the same
    if loop_dict.keys() != hue_dict.keys():
        raise ValueError("The keys of the loop_dict and hue_dict must be the same.")
    # If hue_dict values are not covered in color_dict's keys
    if set(hue_dict.values()) - set(color_dict.keys()):
        raise ValueError("The values of the hue_dict must be covered in the color_dict's keys.")

    # If complete_threshold is not between 0 and 1
    if complete_threshold < 0 or complete_threshold > 1:
        raise ValueError("complete_threshold should be between 0 and 1")

    # Create a dictionary to hold the results
    results = {}
    # Loop over the groups
    for group, samples in loop_dict.items():
        cur_subset = data[samples]
        # Calculate number of proteins with at least
        results[group] = cur_subset[(
            ((~(cur_subset.isna())).sum(axis=1) / len(samples)) >= complete_threshold
        )].shape[0]

    # Create a dataframe for plotting data completeness
    plot_data = pd.concat(
        [
            pd.Series(hue_dict, name=hue_name),
            pd.Series(results, name=feature_name)
        ], 
        axis=1
    ).sort_values(
        [hue_name, feature_name], ascending=[True, False]
    )

    df_min = plot_data[feature_name].min()
    df_med = plot_data[feature_name].median()

    plt.figure(figsize=(18, 6))
    # Use bar plot from matplotlib
    plt.bar(
        x=range(plot_data.shape[0]),
        height=plot_data[feature_name],
        color=plot_data[hue_name].map(color_dict),
        edgecolor="white",
        linewidth=0.1
    )
    # Remove X-axis labels
    plt.xticks([])
    # Add X-axis labels
    plt.xlabel("")
    # Add Y-axis labels
    plt.ylabel(feature_name, fontsize=18)

    # Add legend 
    #  - Create a list of patches
    patches = []
    #  - Loop over the tissue types
    for hue_var, color in color_dict.items():
        #  - Create a patch for each tissue type
        patches.append(mpatches.Patch(color=color, label=hue_var))
    #  - Add the legend to the bottom of the plot with 10 columns

    plt.legend(
        handles=patches, 
        loc="lower center", 
        ncol=7, 
        fontsize=12, 
        bbox_to_anchor=(0.5, -0.375)
    )

    # Add median line with a text label at the end
    plt.axhline(
        y=plot_data[feature_name].median(), 
        color=label_color,
        linestyle="--",
        linewidth=2,
        alpha=0.5
    )
    plt.text(
        x=label_x_offset,
        y=df_med + label_y_offset,
        s="Median: {}".format(df_med),
        fontweight="bold",
        fontsize=12,
        color=label_color,
        verticalalignment="center"
    )

    # Add minimum line with a text label at the end
    plt.axhline(
        y=df_min,
        color=label_color,
        linestyle="--",
        linewidth=2,
        alpha=0.5
    )
    plt.text(
        x=label_x_offset,
        y=df_min + label_y_offset,
        s="Minimum: {}".format(df_min),
        fontweight="bold",
        fontsize=12,
        color=label_color,
        verticalalignment="center"
    )

    # Add total proteins line with a text label at the end
    plt.axhline(
        y=data.shape[0],
        color=label_color,
        linestyle="--",
        linewidth=2,
        alpha=0.5
    )
    plt.text(
        x=label_x_offset,
        y=data.shape[0]+label_y_offset,
        s="Total: {}".format(data.shape[0]),
        fontweight="bold",
        fontsize=12,
        color=label_color,
        verticalalignment="center"
    )

    # Add the fully quantified proteins line with a text label at the end
    plt.axhline(
        y=data.dropna().shape[0],
        color=label_color,
        linestyle="--",
        linewidth=2,
        alpha=0.5
    )
    plt.text(
        x=label_x_offset,
        y=data.dropna().shape[0]+label_y_offset,
        s="Fully quantified: {}".format(data.dropna().shape[0]),
        fontweight="bold",
        fontsize=12,
        color=label_color,
        verticalalignment="center"
    )

    # Add title
    plt.title(
        (
            "Showing Data Completeness" + " of " + group_name + " have at least " +
            str(int(complete_threshold*100)) + "% of their replicates quantified" +
            "\n" + 
            title_add
        ),
        fontsize=20
    )
    # Plot asthetics
    plt.tight_layout()
    sns.despine(left=True, bottom=True)

    # Return the plot object
    return plt

def check_batch_effect(
        pca,
        plot_data: pd.DataFrame,
        color_dict: dict,
        hue_name: str = "Tissue_type",
        style_name: str = "Batch",
        # title_add: str = "",
        figsize: tuple = (10, 6),
        edgeNum: int = 5,
        pointSize: int = 100,
    ):
    """
    """

    # Found the max and min values of the data
    minVal = plot_data.iloc[:, :2].min().min()
    maxVal = plot_data.iloc[:, :2].max().max()

    # Create a PCA plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        x="Dim_1",
        y="Dim_2",
        data=plot_data,
        hue=hue_name,
        palette=color_dict,
        style=style_name,
        s=pointSize,
        edgecolor="black",
        ax=ax
    )

    # Add 0 lines
    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.5
    )

    plt.axvline(
        x=0,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.5
    )

    # # Set the x-axis label with explained variance
    # ax.set_xlabel(
    #     "PC1 - {}%".format(
    #         round(pca.explained_variance_ratio_[0]*100, 2)
    #     )
    # )
    # # Set the y-axis label with explained variance
    # ax.set_ylabel(
    #     "PC2 - {}%".format(
    #         round(pca.explained_variance_ratio_[1]*100, 2)
    #     )
    # )

    # Set a smart limits for x and y axis
    ax.set_xlim(
        minVal - edgeNum, 
        maxVal + edgeNum
    )
    ax.set_ylim(
        minVal - edgeNum, 
        maxVal + edgeNum
    )

    # Move the legend outside the plot
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        title=hue_name,
        prop={"size": 10},
    )

    # Return the plot
    return fig, ax

def single_pair_summary(
        data: pd.DataFrame,
        total_proteins: int,
        pair_names: tuple,
        # Data related parameters
        df_pval: str = "df_p",
        df_qval: str = "df_adjp",
        eq_pval: str = "eq_p",
        eq_qval: str = "eq_adjp",
        log2FC: str = "log2FC",
        logQvalue: str = "log10(adj_pval)",
        status: str = "Status",
        # Stats related parameters
        pThr: float = 0.05, 
        dfThr: float = 1,
        eqThr: float = 1,            
        corr_name: str = "FDR",
        # Plot parameters
        figsize: tuple = (10, 5),
        save: bool = False,
        filename: str = "single_pair_summary",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"], 
        dont_show: bool = False,
    ):
    """

    """

    # Define the status palette
    status_palette = {
        "Different": stat_colors[-2],
        "Equivalent": stat_colors[0],
        "Unexplained": "#99999950",
        "Excluded": stat_colors[2]
    }
    # Check if corr_name is None set as string
    if corr_name is None:
        corr_name = "None"
    
    # Create a counts dataframe 
    cnts = data[status].value_counts()
    cnts["Excluded"] = total_proteins - cnts.sum()
    cnts = cnts.reset_index().rename(
        columns={
            "index": "Status",
            "Status": "Count"
        }
    )

    # Initialize the figure
    fig = plt.figure(
        figsize=figsize
    )
    grid = gridspec.GridSpec(
        nrows=2, 
        ncols=3, 
        width_ratios=[
            0.3, 
            0.6, 
            0.1
        ], 
        wspace=0.2,
        hspace=0.3
    )

    # Initialize the axes
    ax1 = plt.subplot(grid[0, 0]) # T-test pvalue dist
    ax2 = plt.subplot(grid[1, 0]) # TOST pvalue dist
    ax3 = plt.subplot(grid[:, 1:2]) # Antlers Plot
    ax4 = plt.subplot(grid[:, 2]) # Protein Status Counts

    # Plot T-test Histogram of P- & Adj.P-values
    sns.histplot(
        data=data,
        x=df_pval,
        ax=ax1,
        color=stat_colors[-1],
        label="P-Value",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    sns.histplot(
        data=data,
        x=df_qval,
        ax=ax1,
        color=stat_colors[-2],
        label="Adj.P-Value ( " + corr_name + " )",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    ax1.set_title(
        "T-Test (Difference)", 
        y=1, 
        fontsize=12
    )
    ax1.legend(
        loc="upper right",
        frameon=False,
        # title=""
    )
    # Add styling to the plot
    ax1.set_xlim([0, 1])
    ax1.grid(True)
    ax1.set_xlabel("")
    ax1.set_ylabel("Frequency")
    ax1.set_xticklabels([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    # ax1.spines["left"].set_visible(False)

    sns.histplot(
        data=data,
        x=eq_pval,
        ax=ax2,
        color=stat_colors[1],
        label="P-Value",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    sns.histplot(
        data=data,
        x=eq_qval,
        ax=ax2,
        color=stat_colors[0],
        label="Adj.P-Value ( " + corr_name + " )",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    ax2.set_title(
        "TOST (Equivalence)", 
        y=1, 
        fontsize=12
    )
    ax2.legend(
        loc="upper right",
        frameon=False,
        # title=""
    )
    ax2.set_xlim([0, 1])
    ax2.grid(True)
    ax2.set_xlabel("P-Value")
    ax2.set_ylabel("Frequency")
    # ax2.set_xticklabels([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # ax2.spines["bottom"].set_visible(False)
    # ax2.spines["left"].set_visible(False)

    # Plot Mutant Volcano Plot
    sns.scatterplot(
        data=data,
        x=log2FC,
        y=logQvalue,
        hue=status,
        ax=ax3,
        palette=status_palette,
        # alpha=0.5,
        s=100,
        linewidth=0.5,
        edgecolor="white",
        rasterized=True,
    )
    ax3.set_title(
        "Antlers Plot", 
        y=1, 
        fontsize=12
    )
    ax3.set_xlabel("log2FC")
    ax3.set_ylabel(
        "log10(Adj.P-value)",
        labelpad=-5
    )
    ax3.legend(
        loc="lower left",
        frameon=False,
        # title=""
    ).remove()
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Add lines to the plot to indicate the thresholds
    # For the Adj. P-value threshold (pThr)
    ax3.axhline(
        y=-np.log10(pThr),
        color="#99999975",
        linestyle="--",
        linewidth=1.5,
    )
    ax3.axhline(
        y=np.log10(pThr),
        color="#99999975",
        linestyle="--",
        linewidth=1.5,
    )

    # For T-test (dfThr)
    ax3.axvline(
        x=dfThr,
        color=stat_colors[-1],
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
    )
    ax3.axvline(
        x=-dfThr,
        color=stat_colors[-1],
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
    )

    # For TOST (eqThr)
    ax3.axvline(
        x=eqThr,
        color=stat_colors[1],
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
    )
    ax3.axvline(
        x=-eqThr,
        color=stat_colors[1],
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
    )

    # Plot Protein Status Count Plot
    sns.barplot(
        data=cnts,
        x="Count",
        y="Status",
        ax=ax4,
        palette=status_palette,
        rasterized=True,
        order=[
            "Excluded",
            "Different", 
            "Unexplained", 
            "Equivalent"
        ]
    )
    # Add the counts to the plot
    for p in ax4.patches:
        width = p.get_width()
        if width > 0:
            width = int(width)
        else:
            width = 0
        ax4.text(
            width + 0.15,
            p.get_y() + p.get_height() / 2,
            width,
            ha="left",
            va="center",
            fontsize=12,
            color="k",
        )

    ax4.set_title(
        "Protein Count per Status", 
        y=1, 
        fontsize=12
    )
    ax4.set_xlabel("Protein Count")
    ax4.set_xticks([])
    ax4.set_xticklabels([])
    ax4.set_ylabel("")
    # Rotate the y-axis ticklabels to be parallel to the y-axis
    ax4.set_yticklabels(
        ax4.get_yticklabels(), 
        rotation=90, 
        horizontalalignment="center", 
        verticalalignment="center", 
        fontsize=12
    )
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["bottom"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    
    fig.suptitle(
        (
            "Single Pair's Test Summary ("+ 
            pair_names[0] + 
            " vs " + 
            pair_names[1] + 
            ")"
        ),
        fontsize=14, 
        y=.95,
    )

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename+"_"+pair_names[0]+"_vs_"+pair_names[1],
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close(fig)

def single_pair_tost_pval_distributions(
        data: pd.DataFrame,
        pair_names: tuple,
        # Data related parameters
        eq_lower_pval: str = "eq_lp",
        eq_upper_pval: str = "eq_up",
        eq_lower_pval_adj: str = "eq_ladjp",
        eq_upper_pval_adj: str = "eq_uadjp",
        # Stats related parameters
        pThr: float = 0.05,
        eq_lower_bound: float = -0.5,
        eq_upper_bound: float = 0.5,
        corr_name = "FDR",
        # Plot parameters
        figsize: tuple = (10, 4),
        color_pal: list = ["black", "red"],
        save: bool = False,
        filename: str = "tost_pval_distributions",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ):
    """

    """

    # Initialize a figure
    fig, (ax1, ax2)= plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize, 
        sharex=True, 
        sharey=True, 
        gridspec_kw={
            'wspace': 0.075
        }
    )

    # Plot Test against lower bound P- & Adj.P-values
    sns.histplot(
        data=data,
        x=eq_lower_pval,
        ax = ax1,
        color=color_pal[0],
        label="P-Value",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    sns.histplot(
        data=data,
        x=eq_lower_pval_adj,
        ax = ax1,
        color=color_pal[1],
        label="Adj.P-Value ( " + corr_name + " )",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )

    # Add styling to the plot
    ax1.set_xlim([0, 1])
    ax1.grid(True)
    ax1.set_xlabel(
        "P-Value(Lower Bound = {})".format(
            eq_lower_bound
        )
    )
    ax1.set_ylabel("Frequency")
    # ax1.set_xticklabels([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    # Plot Test against upper bound P- & Adj.P-values
    sns.histplot(
        data=data,
        x=eq_upper_pval,
        ax = ax2,
        color=color_pal[0],
        label="P-Value",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )
    sns.histplot(
        data=data,
        x=eq_upper_pval_adj,
        ax = ax2,
        color=color_pal[1],
        label="Adj.P-Value ( " + corr_name + " )",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
    )

    ax2.legend(
        loc="upper right",
        frameon=False,
        # title=""
    )

    # Add styling to the plot
    ax2.set_xlim([0, 1])
    ax2.grid(True)
    ax2.set_xlabel(
        "P-Value(Upper Bound = {})".format(
            eq_upper_bound
        )
    )
    # ax2.set_ylabel("Frequency")
    # ax2.set_xticklabels([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    fig.suptitle(
        (
            "Equivalence Test P-Value Distributions (" + 
            pair_names[0] + 
            " vs " + 
            pair_names[1] + 
            ")"
        ),
        y=.95,
        fontsize=14, 
    )

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename+"_"+pair_names[0]+"_vs_"+pair_names[1],
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close(fig)

def single_pair_proteins_indepth(
        data: pd.DataFrame,
        protein_data: pd.DataFrame,
        pair_names: tuple,
        # Data related parameters
        protein_col: str = "Protein",
        status_col: str = "Status",
        rank_col: str = "Rank",
        mean_col: str = "Mean",
        # Stats related parameters
        bin_size: int = 500,
        # Plot related parameters
        figsize: tuple = (12, 8),
        xMargin: int = 1500,
        yMargin: float = 1.5,
        save: bool = False,
        filename: str = "single_pair_proteins_Indepth",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ):

    """
    """

    status_labels = {  
        1 : "Retained",
        0 : "Missing", 
        -1: "Filtered"
    }

    status_palette = {
        "Different": "#70161e",
        "Equivalent": "#29335c",
        "Unexplained": "#b1a7a6",
        "Excluded": "#565d61"
    }
    pos_labels = list(status_palette.keys())

    # Build the rank data for first 2 plots
    rank_data = protein_data.copy()
    rank_data = rank_data.join(
        data.set_index([protein_col])[status_col]
    )

    # Get the counts of each status
    cnt_data = data[status_col].value_counts()
    # Reindex cnt_data to include all pos_labels
    cnt_data = cnt_data.reindex(
        pos_labels
    ).fillna(0).astype(int)

    # Build the data for the heatmap
    plot_data = data[["S1_Status", "S2_Status", status_col]].copy()
    plot_data["S1_Status"] = plot_data["S1_Status"].map(status_labels)
    plot_data["S2_Status"] = plot_data["S2_Status"].map(status_labels)
    plot_data = plot_data.pivot_table(
        index="S1_Status", 
        columns="S2_Status", 
        values=status_col, 
        aggfunc="count"
    ).fillna(0).astype(int)
        # If any values are missing in the index or columns, add them with 0
    for i in status_labels.values():
        if i not in plot_data.index:
            plot_data.loc[i] = 0
        if i not in plot_data.columns:
            plot_data[i] = 0
    plot_data = plot_data.loc[
        # Apply custom order
        ["Retained", "Missing", "Filtered"], 
        ["Retained", "Missing", "Filtered"]
    ]

    # Criteria data
    crit_data = pd.DataFrame(
        {
            "All Proteins": (
                cnt_data / 
                cnt_data.sum()
            )*100,
            "Tested Only": (
                cnt_data[["Different", "Unexplained", "Equivalent"]] / 
                cnt_data[["Different", "Unexplained", "Equivalent"]].sum()
            )*100,
            "Significant Only": (
                cnt_data[["Different", "Equivalent"]] / 
                cnt_data[["Different", "Equivalent"]].sum()
            )*100,
        }
    ).loc[pos_labels].T

    # Initialize the figure
    fig = plt.figure(
        figsize=figsize
    )
    # Create a grid for the plots
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        wspace=0.2,
        hspace=0.3, 
        width_ratios=[
            0.4,
            0.6
        ],
        height_ratios=[
            .6,
            .4,
        ]
    )
    # Create the axes for the plots
    ax1 = fig.add_subplot(gs[0, 0]) 
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    # Plot 1 - Protein Status Distribution
    crit_data.plot(
        kind="bar", 
        stacked=True, 
        ax=ax1,
        color=status_palette, 
        edgecolor="white",
        linewidth=0.5,
        width=0.8,
        rot=0,
        legend=False,
    )

    # Rewrite y-ticks labels as percentages
    ax1.set_yticklabels(
        [
            "{:.0f}%".format(x) for x in ax1.get_yticks()
        ]
    )
    # Set Labels 
    ax1.set_xlabel("")
    ax1.set_ylabel("Percentage of Proteins")
    ax1.set_title("Protein Status Distribution (%)")
    # Remove the spines
    sns.despine(
        ax=ax1, 
        bottom=True, 
        # left=True,
        right=True,
        top=True
    )

    # Plot 2 - Protein Retained & Excluded Counts in Detailed
    sns.heatmap(
        data=plot_data,
        ax=ax2,
        fmt='d',
        annot=True,
        annot_kws={"size": 14},
        square=True,
        linewidths=0.75,
        linecolor='white',
        # Remove the colorbar
        cbar=False, 
        cmap="Greys",
        cbar_kws={'shrink': 0.5},
    )

    # Set title
    ax2.set_title(
        "Protein Exclusion Matrix",
    )

    # Add rectangle
    ax2.add_patch(mpatches.Rectangle(
        (0, 0), 
        1, 
        1, 
        fill=False, 
        edgecolor="#ffd60a", 
        lw=5
    ))
    ax2.set_xlabel(
        "Sample: {0}".format(pair_names[-1])
    )
    # ax4.xaxis.tick_top()
    ax2.set_xticklabels(
        ax2.get_xticklabels(), 
        rotation=0
    )
    ax2.set_ylabel(
        "Sample: {0}".format(pair_names[0])
    )
    ax2.set_yticklabels(
        ax2.get_yticklabels(), 
        rotation=0
    ) 

    # Create a stacked bar chart with single x and stacks are value_counts of status
    sns.histplot(
        x=rank_col,
        hue=status_col,
        data=rank_data,
        multiple="stack",
        ax=ax3, 
        palette=status_palette, 
        hue_order=[
            "Excluded", 
            "Different", 
            "Unexplained", 
            "Equivalent"
        ],
        edgecolor="white",
        linewidth=0.5,
        binwidth=bin_size, 
        legend=False,
    )
    # Set x-axis label
    ax3.set_xlabel("Rank of Proteins")
    ax3.set_ylabel("Protein Count (Bin Size = {0})".format(bin_size))
    ax3.set_title("Rank Distribution of Proteins with Status")    
    # Remove the spines
    sns.despine(
        ax=ax3, 
        bottom=True, 
        # left=True,
        right=True,
        top=True
    )

    sns.scatterplot(
        ax=ax4,
        y=mean_col,
        x=rank_col,
        hue=status_col,
        data=rank_data,
        palette=status_palette,
        # Extra parameters
        s=75,
        edgecolor="white",
        linewidth=0.5,
        legend=True,
        alpha=0.75,
        rasterized=True,  
    )

    # Add rugplot only for equivalent
    sns.rugplot(
        data=rank_data[
            rank_data[status_col]=="Equivalent"
        ],
        x=rank_col,
        y=mean_col,
        color=status_palette["Equivalent"],
        alpha=0.5,
        ax=ax4,
        rasterized=True,
        height=0.1,
        legend=False
    )
    # Add rugplot only for different
    sns.rugplot(
        data=rank_data[
            rank_data[status_col]=="Different"
        ],
        x=rank_col,
        y=mean_col,
        color=status_palette["Different"],
        alpha=0.5,
        ax=ax4,
        rasterized=True,
        height=0.05,
        legend=False
    )

    ax4.legend(
        loc="upper right",
        frameon=False,
        title="Protein Status"
    )
    # Set x-axis label
    ax4.set_xlabel("Rank of Proteins")
    ax4.set_ylabel("Mean Intensity of Proteins")
    ax4.set_title("Protein Rank vs Mean Intensity")
    ax4.set_xlim(-xMargin, rank_data.shape[0]+xMargin)
    ax4.set_ylim(
        rank_data[mean_col].min()-yMargin, 
        rank_data[mean_col].max()+yMargin
    )

    sns.despine(
        ax=ax4, 
        bottom=True, 
        # left=True,
        right=True,
        top=True
    )

    fig.suptitle(
        (
            "In depth Protein Status Summary (" + 
            pair_names[0] + 
            " vs " + 
            pair_names[1] + 
            ")"
        ),
        y=.95,
        fontsize=14, 
    )

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename+"_"+pair_names[0]+"_vs_"+pair_names[1],
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close(fig)

# --------------------------- Single Protein Related Plots --------------------------- #

def single_protein_pairwise_fc_heatmap(
        fc_matrix: pd.DataFrame,        
        color_df: pd.DataFrame,
        cbar_label: str = "Absolute Log2 Fold Change",
        dendrogram_ratio: float = 0.1,
        eq_boundary: float = 0.81,
        figsize: tuple = (8, 8),
        cmap = "RdBu_r",
        cluster_method: str = "average",
        cluster_metric: str = "euclidean",
        plot_title = "Pairwise Fold Change Heatmap",
        # Show/Save parameters
        save: bool = False,
        filename: str = "pairwise_fc_heatmap",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ) -> sns.clustermap:
    """
    """

    # Calculate vmax and vmin from eq_boundary (- half , + half)
    vmax = eq_boundary * 2
    vmin = eq_boundary * 0

    # 
    g = sns.clustermap(
        figsize=figsize,
        data=fc_matrix,
        row_colors=color_df,
        col_colors=color_df,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        # Add dendogram
        row_cluster=True,
        col_cluster=True,
        row_linkage=None,
        col_linkage=None,
        method=cluster_method,
        metric=cluster_metric,
        dendrogram_ratio=dendrogram_ratio,
        rasterized=True,
        # linewidths=0.5,
    )

    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.set_title(
        plot_title,
        fontsize=14, 
        fontweight="bold",
        pad=20,
        y=1.15
    )
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.tick_params(
        axis='both',
        which='both',
        length=0
    )

    # Add the colorbar
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=12,
        length=0
    )
    cbar.ax.set_ylabel(
        cbar_label,
        fontsize=10,
        fontweight="normal",
        labelpad=10
    )

    # Move the colorbar
    cbar.ax.set_position([
        g.ax_heatmap.get_position().x1+0.01,
        g.ax_heatmap.get_position().y0,
        0.025,
        g.ax_heatmap.get_position().height
    ])

    # Save the figure
    if save:
        save_figures(
            g,
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close(g)


# ----------------------------- Enrichment Related Plots ----------------------------- #

def enrich_distribution(
        data: pd.DataFrame,
        # Data related parameters
        subset_col: str = "p_value",
        subset_thr: float = 0.05,
        plot_cols: list[str] = ["p_value", "GeneRatio"],
        annot_thr: list[float] = [0.05, None],
        # Plot related parameters
        figsize: tuple = (4, 4),
        horizontal: bool = False,
        pad: float = 0.1,
        bins: int = 50,
        percentiles: list[float] = [0.90, 0.95, 0.99],
        sub_titles: list[str] = ["P-Value", "Gene Ratio"],
        fig_title: str = "Enrichment Results Distribution",
        # Styling parameters
        bin_color=stat_colors[4], 
        bin_edgecolor=stat_colors[2], 
        bin_linewidth=1, 
        bin_alpha=1,
        percentile_color=stat_colors[0], 
        percentile_linestyle='--', 
        percentile_linewidth=1,
        percentile_linelength=0.5, 
        percentile_alpha=1, 
        percentile_fontsize=12,
        percentile_horizontalalignment='right', 
        percentile_verticalalignment='center',
        percentile_rotation=90,
        threshold_color=stat_colors[5], 
        threshold_linestyle='--', 
        threshold_linewidth=1,
        threshold_alpha=1, 
        threshold_fontsize=12, 
        threshold_horizontalalignment='right',
        threshold_verticalalignment='center', 
        threshold_rotation=90,
        # Show/Save parameters
        save: bool = False,
        filename: str = "enrich_results_histogram",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ):  
    # TODO: Add docstring
    """
    
    """
    # TODO: Add more sanity checks for the inputs
    ## Sanity Checks 
    # Check if plot_cols and thresholds are the same length
    if len(plot_cols) != len(annot_thr):
        raise ValueError(
            """plot_cols and thresholds must be the same length in the same order.
            If wants to plot some columns without threshold use None as placeholder."""
        )
    else:
        # Compose dictionary of columns and thresholds
        plot_dict = dict(zip(plot_cols, annot_thr))

    # Check if the plot_cols exist in the data
    for col in plot_cols:
        if col not in data.columns:
            raise ValueError(
                "Column {0} not in the data.".format(col)
            )
    # Check if subset_col exist in the data
    if subset_col is not None:
        if subset_col not in data.columns:
            raise ValueError(
                "Column {0} not in the data.".format(subset_col)
            )
        else:
            # Subset the data
            thr_data = data[data[subset_col] <= subset_thr]

    # Based on the users input decide the direction of multiple plots
    if horizontal:
        nrows = 1 
        ncols = len(plot_cols)
        sharex = False
        sharey = True

    else:
        nrows = len(plot_cols)
        ncols = 1
        sharex = True
        sharey = False

    # Initialize the figure
    fig, ax = plt.subplots(
        nrows = nrows,
        ncols = ncols,
        figsize = figsize,
        sharex = sharex,
        sharey = sharey,
        # Add pad
        gridspec_kw = {
            "wspace": pad,
            "hspace": pad
        }
    )
    # Loop through plot columns
    for i, col in enumerate(plot_cols):
        if len(plot_cols) == 1:
            # If only one column to plot
            ax = [ax]

        if sub_titles[i] is not None:
            cur_sub_title = sub_titles[i]
        else: 
            cur_sub_title = col.replace("_", " ").title()

        # Plot the histogram of the data
        sns.histplot(
            data=data,
            x=col,
            ax=ax[i],
            bins=bins,
            color=bin_color,
            edgecolor=bin_edgecolor,
            linewidth=bin_linewidth,
            alpha=bin_alpha,
        )

        # Add subplot styling
        ax[i].set_xlabel(
            cur_sub_title
        )
        ax[i].set_ylabel(
            "Frequency (total terms: {0})".format(
                data.shape[0]
            )
        )

        # Add vertical lines for percentiles
        for p in percentiles:
            # if thr_data is created
            if subset_col is not None:
                # Get the percentile value
                xcor = thr_data[col].quantile(p)
            else:
                # Get the percentile value
                xcor = data[col].quantile(p)

            # Add the vertical line
            ax[i].axvline(
                x=xcor,
                color=percentile_color,
                linestyle=percentile_linestyle,
                linewidth=percentile_linewidth,
                alpha=percentile_alpha,
                ymax=percentile_linelength,
            )

            # Add annotation to the plot
            ax[i].text(
                x=xcor,
                y=0.37,
                s=str(int(p*100)) + "th percentile = " + str(round(xcor, 1)),
                color=percentile_color,
                fontsize=percentile_fontsize,
                horizontalalignment=percentile_horizontalalignment,
                verticalalignment=percentile_verticalalignment,
                rotation=percentile_rotation,
                transform=ax[i].get_xaxis_transform(), 
            )

        # Add annot_thr if not None
        if plot_dict[col] is not None:
            # Add cutoff line to the plot
            ax[i].axvline(
                x=plot_dict[col],
                color=threshold_color,
                linestyle=threshold_linestyle,
                linewidth=threshold_linewidth,
                alpha=threshold_alpha,
            )
            # Add text highlighting the cutoff
            ax[i].text(
                x=plot_dict[col],
                y=0.5,
                s="Threshold: " + str(subset_thr),
                color=threshold_color,
                fontsize=threshold_fontsize,
                horizontalalignment=threshold_horizontalalignment,
                verticalalignment=threshold_verticalalignment,
                rotation=threshold_rotation,
                transform=ax[i].get_xaxis_transform(), 
            )

        # Set the title of the figure
        ax[i].set_title(
            "Histogram of {0}".format(
                cur_sub_title
            ),
            y=0.90
        )
        # Add x and y grid
        ax[i].grid(
            axis="both",
            alpha=0.75,
            linestyle="dotted",
        )

        # Despine the plot
        sns.despine(
            ax=ax[i],
            left=True,
            bottom=False,
            offset=10,
            trim=True
        )

    # Set the title of the figure
    fig.suptitle(
        fig_title,
        y=.95,
        fontsize=14,
    )

    # Tight layout
    fig.tight_layout()

    # Save the figure
    if save:
        save_figures(
            fig,
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close(fig)

def enrich_terms_upset(
        data: pd.DataFrame,
        # Data parameters
        main_id_col: str = "query",
        feature_col: str = "native",
        secondary_id_col: str = "source",
        concat_str: str = " | ",
        # Plot parameters
        stacked: bool = True,
        subset_size: str= "count",
        show_counts: bool = True,
        min_degree: int = None,
        min_subset_size: int = None,
        orientation: str = "vertical",
        sort_by: str = "cardinality",
        sort_categories_by: str = None,
        bars_ylabel: str = "Number of Enriched Terms",
        fig_title: str = "Enriched Terms",
        # Style parameters
        facecolor: str = "darkblue",
        shading_color: str = "lightgray",
        element_size: int = 30,
        custom_palette: dict = None,
        legend_fontsize: int = 8,
        title_fontsize: int = 10,
        title_fontweight: str = "bold",
        title_height: float = 0.95,
        # Save parameters
        save: bool = False,
        filename: str = "enrich_terms_upsetplot",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ):
    """
        Draws an upset plot from the enrichment data, 
            if data has multiple enrichment sources, 
            it can be used as in a stacked or grouped manner.
    """
    if custom_palette is None:
        custom_palette = enrichment_palette
    if stacked:
        intersection_plot_elements = 0
    else:
        intersection_plot_elements = 1
    
    # Pass variables to the make_upset_data function
    plot_data = utils.make_upset_data(
        data = data,
        stacked = stacked,
        main_id_col = main_id_col,
        feature_col = feature_col,
        secondary_id_col = secondary_id_col,
        concat_str = concat_str,
    )

    # Create the main Upset Figure
    upset = UpSet(
        plot_data,
        subset_size = subset_size,
        show_counts = show_counts,
        sort_by = sort_by,
        sort_categories_by = sort_categories_by,
        orientation = orientation,
        facecolor = facecolor,
        shading_color = shading_color,
        element_size = element_size,
        min_degree = min_degree,
        min_subset_size = min_subset_size,
        intersection_plot_elements = intersection_plot_elements
    )
    # Add the stacked bars
    upset.add_stacked_bars(
        by=secondary_id_col, 
        colors=custom_palette,
        title=bars_ylabel,
        elements=10
    )

    # Get axes as dictionary
    axes = upset.plot()

    # Place the legend outside the plot
    if orientation == "vertical":
        # If vertical place on top
        axes["extra0"].legend(
            loc="upper center",
            bbox_to_anchor=(
                0.5, 1.25 # TODO: Needs to scale with the plot
            ),
            ncol=len(enrichment_palette),
            frameon=False, 
            fontsize=legend_fontsize
        )
    else:
        # If horizontal place on left side
        axes["extra0"].legend(
            loc="center left",
            bbox_to_anchor=(
                -1.5, 0.85 # TODO: Needs to scale with the plot
            ),
            # ncol=len(enrichment_palette),
            frameon=False,
            fontsize=legend_fontsize
        )

    # Suptitle
    plt.suptitle(
        fig_title,
        fontsize=title_fontsize,
        fontweight=title_fontweight,
        y=title_height
    )
    plt.tight_layout()
    # Save the figure
    if save:
        save_figures(
            plt.gcf(),
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close()

def enrich_term_counts(
        data: pd.DataFrame,
        queries: list[str] = None,
        sources: list[str] = None,
        # Data parameters
        query_col: str = "query",
        source_col: str = "source",
        info_cols: list[str] = ["native"],
        val_col: str = "p_value",
        perspective: str = "group-specific",
        group_combination: str = "all",
        # Plot parameters
        figsize: tuple = (6, 6),
        orientation: str = "horizontal",
        annotate: str = None, # None, "total", "group"
        min_count: int = 0,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        legend_title: str = "Enrichment Source",
        legend_xypos = (0.5, 1.025),
        # Style parameters
        custom_palette: dict = None,
        bar_width: float = 0.8,
        bar_linewidth: float = 0.5,
        bar_edgecolor: str = "white",
        annot_rotation: int = 0,
        annot_fontsize: int = 8,
        annot_fontweight: str = "bold",
        annot_lightColor: str = "#eeeeee",
        annot_darkColor: str = "#333333",
        annot_xoffset: int = 0,
        annot_yoffset: int = 0,
        title_fontsize: int = 10,
        title_fontweight: str = "bold",
        title_height: float = 0.95,
        # Save parameters
        save: bool = False,
        filename: str = "enrich_term_counts",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ):
    """
        Visualize enriched term counts over groups and different sources of enrichment.
    """

    # Check if query_col is in the data
    if query_col not in data.columns:
        raise ValueError(
            "Column {0} not in the data.".format(query_col)
        )
    # Check if source_col is in the data
    if source_col not in data.columns:
        raise ValueError(
            "Column {0} not in the data.".format(source_col)
        )
    # if queries is None, get the unique queries from the data
    if queries is None:
        queries = sorted(
            data[query_col].unique().tolist()
        )
    # if sources is None, get the unique sources from the data
    if sources is None:
        sources = sorted(
            data[source_col].unique().tolist()
        )

    # If the query is not a list, raise an error
    if not isinstance(queries, list):
        raise TypeError("The query must be a list of strings.")
    # If the query is empty, raise an error
    if len(queries) == 0:
        raise ValueError("The query cannot be empty.")
    # If the query is a list of one element, raise an error
    if len(queries) == 1:
        print("Single query: {0} passed! Will only use this for plotting. Check if it is a mistake.".format(queries[0]))
    # If the perspective is not one of the allowed values, raise an error
    if perspective not in [
            "groups",
            "group-specific", 
            "term-occurrence"
        ]:
        raise ValueError(
            """
            The perspective must be one of the following: 
            ['groups', 'group-specific', 'term-occurrence']
            """
        )

    # If custom_palette is not provided, use the default palette
    if custom_palette is None:
        custom_palette = enrichment_palette

    # If the group_combination is not one of the allowed values, raise an error
    if group_combination not in ["all", "simple"]:
        raise ValueError(
            "The group_combination must be one of the following: 'all', 'simple'"
        )
    # Set the kind of the plot based on the orientation
    if orientation == "horizontal":
        kind = "barh"
    elif orientation == "vertical":
        kind = "bar"
    else: # If the orientation is not one of the allowed values, raise an error
        raise ValueError(
            "The orientation must be one of the following: 'horizontal', 'vertical'"
        )

    # Create a Completeness DataFrame (Wide Format)
    completeness_df = (
        ~data.pivot(
            index=[source_col] + info_cols,
            columns=query_col,
            values=val_col
        ).isna()
    )

    if perspective == "group-specific":
        # Add couple of ValueError checks
        if (len(queries) > 4) and (group_combination == "all"):
            raise ValueError(
                """Exhaustive combinations of groups are not allowed for more than 4 groups.
                You can use group_combination = 'simple', to allow plotting the following:
                    'shared in all', 'shared in some', and distinct groups
                """
            )
        # If the group_combination is 'all', create all possible combinations of groups
        if group_combination == "all":
            # Create specific title, xlabel, and ylabel if not provided
            if title is None:
                title = "Enriched Term Counts per All Group Combinations"
            if xlabel is None:
                xlabel = "Number of Enriched Terms"
            if ylabel is None:
                ylabel = "All Group Combinations"

            combs = []
            for i in range(1, len(queries)+1):
                combs.extend(list(combinations(queries, i)))

            cur_dict = {}
            for comb in combs:
                cur_name = " &\n".join(comb) # TODO: Add a parameter to change the separator
                tmp = completeness_df.loc[:, comb]
                tmp = (
                    (tmp.all(axis=1)) & 
                    (completeness_df.sum(axis=1) == len(comb))
                )
                cur_dict[cur_name] = tmp[
                    tmp == True
                ].reset_index().value_counts(
                    source_col
                ).loc[sources].to_dict()
            
            # Create a DataFrame from the dictionary
            plot_data = pd.DataFrame(
                cur_dict
            ).fillna(0).astype(int).T
        # If the group_combination is 'simple', create simple combinations:
        #   shared in all, shared in some, and distinct groups
        if group_combination == "simple":
            # Create specific title, xlabel, and ylabel if not provided
            if title is None:
                title = "Enriched Term Counts per Simplified Group Combinations"
            if xlabel is None:
                xlabel = "Number of Enriched Terms"
            if ylabel is None:
                ylabel = "Simpliefied Group Combinations"

            # Create a custom mapper 
            mapper = {}
            for i in range(0, completeness_df.shape[1]+1):
                if i == 0:
                    val = "Not Enriched"
                elif i == 1:
                    val = "Distinct"
                elif i == completeness_df.shape[1]:
                    val = "Shared in All"
                else:
                    val = "Shared in Some"
                mapper[i] = val

            # Find occurrences of each term and map them to the custom labels
            occ_data = completeness_df.sum(axis=1).replace(mapper)
            # Create a copy of the data and find distinct occurrences actual labels
            acc_data = data.copy().set_index([source_col]+info_cols)[query_col]
            # Replace the occ_data's "Distinct" label with actual label from acc_data
            occ_data[occ_data == "Distinct"] = acc_data[occ_data == "Distinct"]
            # Create a DataFrame from the occ_data
            plot_data = occ_data.reset_index().groupby([
                source_col, 0, 
            ]).size().unstack().fillna(0).astype(int).T
    
    # Shows number of terms per their occurrence in queries 0, 1, 2, 3...
    if perspective == "term-occurrence":
        # Create specific title, xlabel, and ylabel if not provided
        if title is None:
            title = "Enriched Term Counts per Occurrence"
        if xlabel is None:
            xlabel = "Number of Enriched Terms"
        if ylabel is None:
            ylabel = "Occurrence of Terms (N = Shared in N Groups)"

        plot_data = completeness_df.sum(axis=1).reset_index().groupby([
            source_col, 0,
        ]).size().unstack().fillna(0).astype(int).T

    if perspective == "groups":
        # Create specific title, xlabel, and ylabel if not provided
        if title is None:
            title = "Enriched Term Counts per Group"
        if xlabel is None:
            xlabel = "Number of Enriched Terms"
        if ylabel is None:
            ylabel = "Groups"

        plot_data = completeness_df.reset_index().groupby(source_col).sum().T
        plot_data = plot_data.reindex(
            index=queries,
            fill_value=0
        ).loc[queries[::-1]]

    # Initialize the figure
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=figsize,
        # sharex=True,
        # sharey=True
    )
    # Plot the data
    plot_data.plot(
        kind=kind,
        stacked=True,
        color=custom_palette,
        edgecolor=bar_edgecolor,
        linewidth=bar_linewidth,
        legend=True,
        width=bar_width,
        ax=ax,
    )
    # Annotate the plot if user specified
    if annotate is not None:
        # Annotate the total counts 
        if annotate == "total":
            for i, v in enumerate(plot_data.sum(axis=1)):
                if kind == "barh":
                    xcor = v + annot_xoffset
                    ycor = i + annot_yoffset
                    ha  = "left"
                    va = "center"
                if kind == "bar":
                    xcor = i + annot_xoffset
                    ycor = v + annot_yoffset
                    ha  = "center"
                    va = "bottom"
                ax.text(
                    xcor, 
                    ycor,
                    str(int(v)),
                    ha=ha,
                    va=va,
                    rotation=annot_rotation,
                    fontsize=annot_fontsize,
                    fontweight=annot_fontweight,
                    color=pick_color_based_on_background(
                        ax.get_facecolor(), 
                        lightColor=annot_lightColor,
                        darkColor=annot_darkColor,
                        uicolor=True
                    )
                )
        # Annotate the group counts
        elif annotate == "group":
            for p in ax.patches:
                total_height = 0
                width, height = p.get_width(), p.get_height()
                total_height += height
                x, y = p.get_xy()
                # if the bar is too small, don't annotate
                
                if kind == "barh":
                    if width <= min_count:
                        continue
                    xcor = x + width / 2
                    ycor = y + height / 2
                    ha  = "center"
                    va = "center"
                    val = str(int(width))
                if kind == "bar":
                    if height <= min_count:
                        continue
                    xcor = x + width / 2
                    ycor = y + total_height / 2
                    ha  = "center"
                    va = "center"
                    val = str(int(height))
                ax.text(
                    xcor, 
                    ycor,
                    val,
                    ha='center',
                    va='center',
                    rotation=annot_rotation,
                    fontsize=annot_fontsize,
                    fontweight=annot_fontweight,
                    color=pick_color_based_on_background(
                        p.get_facecolor(), 
                        lightColor=annot_lightColor,
                        darkColor=annot_darkColor,
                        uicolor=True
                    )
                )
    
    # Set the x-axis label
    ax.set_xlabel(
        xlabel if kind == "barh" else ylabel
    )
    # Set the y-axis label
    ax.set_ylabel(
        ylabel if kind == "barh" else xlabel
    )
    # Set the title
    ax.set_title(
        title, 
        fontsize = title_fontsize,
        fontweight = title_fontweight,
        y = title_height
    )
    # Set the legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=legend_xypos,
        ncol=len(custom_palette.keys()),
        frameon=False,
        title=legend_title,
    )
    fig.tight_layout()
    sns.despine(
        ax=ax,
        top=True,
        right=True,
        left=True,
        bottom=False,
    )
    # Save the figure
    if save:
        save_figures(
            plt.gcf(),
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close()

def enrich_term_grouped_heatmap(
        data: pd.DataFrame,
        queries: list[str] = None,
        sources: list[str] = None,
        # Data parameters
        query_col: str = "query",
        source_col: str = "source",
        info_cols: list[str] = ["native"],
        quant_col: str = "-log10(p_value)",
        vmin: float = 0,
        prctl: float = 95, 
        # Plot parameters
        figsize: tuple = (12, 3),
        figheight_ratios: list = [0.1, 1],
        fighspace: float = 0.01,
        min_count: int = 0,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        # Style parameters
        bgColor: str = "#adb5bd",
        colorbar: str = "Blues",
        custom_palette: dict = None,
        cbar_label: str = None,
        cbar_orientation: str = "horizontal",
        cbar_shrink: float = 0.5,
        cbar_aspect: float = 50,
        cbar_pad: float = 0.15,
        title_fontsize: int = 10,
        title_fontweight: str = "bold",
        title_height: float = 0.95,
        header_lightColor: str = "#eeeeee",
        header_darkColor: str = "#333333",
        header_rotation: int = 0,
        header_fontsize: int = 8,
        header_fontweight: str = "bold",
        # Save parameters
        save: bool = False,
        filename: str = "enrich_term_grouped_heatmap",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ):
    """
        Visualizes the enriched terms over queries
            and grouped by sources in a identification heatmap.
    """

    # Check if query_col is in the data
    if query_col not in data.columns:
        raise ValueError(
            "Column {0} not in the data.".format(query_col)
        )
    # Check if source_col is in the data
    if source_col not in data.columns:
        raise ValueError(
            "Column {0} not in the data.".format(source_col)
        )
    # if queries is None, get the unique queries from the data
    if queries is None:
        queries = sorted(
            data[query_col].unique().tolist()
        )
    # if sources is None, get the unique sources from the data
    if sources is None:
        sources = sorted(
            data[source_col].unique().tolist()
        )

    # If the query is not a list, raise an error
    if not isinstance(queries, list):
        raise TypeError("The query must be a list of strings.")
    # If the query is empty, raise an error
    if len(queries) == 0:
        raise ValueError("The queries cannot be empty.")
    
    # if the source is not a list, raise an error
    if not isinstance(sources, list):
        raise TypeError("The source must be a list of strings.")
    # If the source is empty, raise an error
    if len(sources) == 0:
        raise ValueError("The sources cannot be empty.")
    
    # If custom_palette is not provided, use the default palette
    if custom_palette is None:
        custom_palette = enrichment_palette

    # Create specific title, xlabel, and ylabel if not provided
    if title is None:
        title = "Comparing Enriched Terms Across Queries and Sources"
    if xlabel is None:
        xlabel = "All Enriched Terms (n={})".format(data[info_cols[0]].nunique())
    if ylabel is None:
        ylabel = ""
    if cbar_label is None:
        cbar_label = quant_col
    

    # Create a plot that that is wide format
    plot_data = data.pivot(
        index=[source_col] + info_cols,
        columns=query_col,
        values=quant_col
    # Add missing queries as columns
    ).reindex(
        columns=queries,
        fill_value=np.nan
    )
    
    # TODO: The sorting should take into account the individual groups
    # Default: sort based on the number of groups and sum of the quant_col 
    orderedIndex = pd.DataFrame(
        {
            "NGroup": (~plot_data.isna()).sum(axis=1),
            "Sum": plot_data.sum(axis=1)
        }
    ).reset_index().sort_values(
        [source_col, "NGroup", "Sum"], 
        ascending=[True, True, True]
    ).set_index([source_col] + info_cols).index

    # Find source_sizes
    source_sizes = plot_data.reset_index().groupby(
        source_col
    # Add missing sources as rows
    ).size().reindex(
        index=sources,
        fill_value=0
    )

    # Reorder the plot_data based on the orderedIndex
    plot_data = plot_data.loc[orderedIndex].T

    # Create a mask for the missing values
    mask = plot_data.isna()

    # Find a specific percentile as a vmax
    vmax = np.nanpercentile(
        plot_data.values,
        prctl
    )

    # Initialize the figure
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        # sharey=True,
        # sharex=True,
        gridspec_kw={
            "height_ratios": figheight_ratios,
            "hspace": fighspace
        },
    )
    # Plot the heatmap
    g = sns.heatmap(
        data=plot_data,
        mask=mask,
        cmap=colorbar,
        vmin=vmin,
        vmax=np.ceil(vmax),
        ax=ax[1],
        cbar_kws={
            "label": quant_col,
            "orientation": cbar_orientation,
            "shrink": cbar_shrink,
            "aspect": cbar_aspect,
            "pad": cbar_pad,
        },
        xticklabels=False, 
        rasterized=True
    )
    g.set_facecolor(bgColor)

    # Create a bar plot indicating the heatmap's columns belonging. 
    # The bars should be stretched to the size of the heatmap's columns
    cur_loc = 0 
    for i, (source, size) in enumerate(source_sizes.items()):
        # Add the bars
        ax[0].barh(
            y=0,
            width=size,
            height=1,
            left=cur_loc,
            color=custom_palette[source],
            edgecolor="white",
            linewidth=0.25,
            label=source
        )
        if size > min_count:
            # Add the text
            ax[0].text(
                x=cur_loc + size / 2,
                y=0.025,
                s=source,
                ha="center",
                va="center",
                color=pick_color_based_on_background(
                    custom_palette[source],
                    lightColor=header_lightColor,
                    darkColor=header_darkColor,
                    hex=True
                ),
                rotation=header_rotation,
                fontsize=header_fontsize,
                fontweight=header_fontweight,
            )
        cur_loc += size

    # Set styling to plots
    ax[0].set_xlim(0, cur_loc)
    ax[0].axis("off")
    ax[0].set_title(
        title,
        fontsize = title_fontsize,
        fontweight = title_fontweight,
        y = title_height
    )
    # Add x and y labels
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)

    # tight layout and despine
    fig.tight_layout()
    sns.despine(
        ax=ax[1],
        left=True
    )

    # Save the figure
    if save:
        save_figures(
            plt.gcf(),
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close()

def enrich_term_clustermap(
        data: pd.DataFrame,
        linkage: np.ndarray,
        clusters: pd.Series,
        # Data parameters
        orientation: str = "vertical",
        number_of_clusters: int = None,
        quant_col: str = "-log10(p_value)",
        # Plot parameters
        figsize: tuple = (5, 5),
        dendrogram_ratio: tuple = (0.1, 0.2),
        # TODO: Make this easily adjustable for orientation
        cbar_pos: tuple = (0.895, 0.15, 0.03, 0.6),
        cbar_label: str = None,
        cbar_min: float = 0,
        cbar_max: float = 1,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        # Style parameters
        values_palette: str = "rocket_r",
        cluster_palette: str = "Greys",
        # Save parameters
        save: bool = False,
        filename: str = "enrich_term_clustermap",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ):
    """
        Takes a pivoted dataframe and plots a clustermap 
            using a custom clustering applied to row or columns.
        data: pd.DataFrame
            A pivoted dataframe with rows as features and columns as queries.
        linkage: np.ndarray
            A linkage matrix for the clustering.
        clusters: pd.Series
            A pandas Series with the cluster labels.
        orientation: str
            The orientation of the clustermap, either "vertical" or "horizontal". 
            Default: "vertical", the features are on the rows and queries on the columns.
    """

    # Check if orientation is one of the allowed values
    if orientation not in ["vertical", "horizontal"]:
        raise ValueError(
            "The orientation must be one of the following: 'vertical', 'horizontal'"
        )
    
    # Check if the linkage is a numpy array
    if not isinstance(linkage, np.ndarray):
        raise TypeError("The linkage must be a numpy array.")
    # Check if the clusters is a pandas Series
    if not isinstance(clusters, pd.Series):
        raise TypeError("The clusters must be a pandas Series.")
    
    # Use default values if not provided
    if cbar_label is None:
        cbar_label = quant_col

    # Get the number of clusters from the row_clusters
    number_of_clusters = clusters.nunique()
    # Create a color palette for the clusters
    cluster_colors = clusters.map(
        lambda x: sns.color_palette(
            cluster_palette,
            number_of_clusters
        )[x-1]
    )

    # Based on the orientation, set the plot parameters
    if orientation == "vertical":
        plot_data = data
        row_cluster = True
        col_cluster = False # TODO: Make this adjustable
        row_linkage = linkage
        col_linkage = None
        row_colors = cluster_colors
        col_colors = None
        xticklabels = True
        yticklabels = False
        if xlabel is None:
            xlabel = ""
        if ylabel is None:
            ylabel = "Enriched Terms"
    elif orientation == "horizontal":
        plot_data = data.T
        row_cluster = True # TODO: Make this adjustable
        col_cluster = True  
        row_linkage = None
        col_linkage = linkage
        row_colors = None
        col_colors = cluster_colors
        xticklabels = False
        yticklabels = True
        if xlabel is None:
            xlabel = "Enriched Terms"
        if ylabel is None:
            ylabel = ""

    # Apply the clustermap
    g = sns.clustermap(
        plot_data,
        figsize=figsize,
        cmap=values_palette,
        cbar_kws={"label": cbar_label},
        metric="euclidean",
        method="ward",
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        row_colors=row_colors,
        col_colors=col_colors,
        xticklabels=xticklabels,
        yticklabels=yticklabels,        
        rasterized=True,
        dendrogram_ratio=dendrogram_ratio,
        # cbar_pos=cbar_pos, # TODO: Make this adjustable
        vmin=cbar_min,
        vmax=cbar_max,
    )

    # Style the plot
    g.ax_heatmap.set_title(title)
    g.ax_heatmap.set_xlabel(xlabel)
    g.ax_heatmap.set_ylabel(ylabel)
    
    # TODO: Make this adjustable
    # if orientation == "vertical":
    #     g.ax_heatmap.xaxis.tick_top()
    #     # g.ax_heatmap.xaxis.set_label_position("top")
    # if orientation == "horizontal":
    #     g.ax_heatmap.yaxis.tick_left()
    #     # g.ax_heatmap.yaxis.set_label_position("left")
    
    # Save the figure
    if save:
        save_figures(
            plt.gcf(),
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close()
    
def enrich_dotplot(
        data: pd.DataFrame,  
        # Data parameters
        term_col: str = "term",
        term_label_col: str = None,
        annotate: bool = True,
        annotate_method: str = 'size',
        query_col: str = "query",
        hue_col: str = "-log10(p_value)",
        size_col: str = "GeneRatio",
        marker_col: str = None, 
        annotate_non_sig: bool = False,
        # Plot parameters
        figsize: tuple = (4, 6),
        hue_pal: str = "Blues",
        hue_norm: tuple = (0, 1),
        sizes: tuple = (10, 100),
        size_norm: tuple = (0, 1),
        apply_tight_layout: bool = True,
        min_count: int = 0,
        color_threshold: float = 0.5,
        # Style parameters
        markers: list = None, 
        marker_alpha: float = 0.95,
        marker_linewidth: float = 0.25,
        marker_edgecolor: str = "white",
        grid_direction: str = "both", # "x", "y", "both"
        grid_color: str = "grey",
        grid_alpha: float = 0.25,
        grid_linestyle: str = "-",
        grid_linewidth: float = 1.,
        xlabel: str = None,
        xlabel_fontsize: int = 8,
        xlabel_fontweight: str = "normal",
        xlabel_flip: bool = False,
        ylabel: str = None,
        ylabel_fontsize: int = 8,
        ylabel_fontweight: str = "normal",
        title: str = None,
        title_fontsize: int = 10,
        title_fontweight: str = "bold",
        title_pos: tuple = (0.0, 1.025),
        title_loc: str = "center",
        xticklabel_fontsize: int = 8,
        xticklabel_rotation: int = 0,
        xtick_flip: bool = False,
        yticklabel_fontsize: int = 8,
        yticklabel_rotation: int = 0,
        legend_xypos: tuple = (1, 1),
        legend_ncol: int = 1,
        legend_title: str = None,
        legend_fontsize: int = 8,
        plot_margins: tuple = (0.2, 0.025),
        annotate_fontsize: int = 8,
        annotate_fontweight: str = "bold",
        annotate_lightColor: str = "#eeeeee",
        annotate_darkColor: str = "#333333",
        # Save parameters
        save: bool = False,
        filename: str = "enrich_dotplot",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ):
    """
        Visualizes the selected enriched terms over queries in a 
            heatmap like dot plot. 
    """
    # Check if term_col is in the data
    if term_col not in data.columns:
        raise ValueError(
            "Column {0} not in the data.".format(term_col)
        )
    # Check if query_col is in the data
    if query_col not in data.columns:
        raise ValueError(
            "Column {0} not in the data.".format(query_col)
        )

    # if hue_col is not in the data, raise an error
    if hue_col not in data.columns:
        raise ValueError(
            "Column {0} not in the data.".format(hue_col)
        )
    # if size_col is not in the data, raise an error
    if size_col not in data.columns:
        raise ValueError(
            "Column {0} not in the data.".format(size_col)
        )

    if term_label_col is None:
        term_label_col = term_col
    else:
        # Check if term_label_col is in the data
        if term_label_col not in data.columns:
            raise ValueError(
                "Column {0} not in the data.".format(term_label_col)
            )

    if annotate_method not in ["hue", "size"]:
        raise ValueError(
            "The annotate_method must be one of the following: 'hue', 'size'"
        )
    else:
        if annotate_method == "hue":
            annotate_col = hue_col
            non_annote_col = size_col
        else:
            annotate_col = size_col
            non_annote_col = hue_col

    if marker_col is None:
        marker = 'o'
        style = None
        markers = None
    else:
        marker = None
        style = marker_col
        if markers is not None:
            markers = {k: markers[k] for k in data[marker_col].unique()}
        else:
            markers = {
                True: 'o',
                False: '.'
            }
        

    # Initialize the figure
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=figsize,
        # sharex=True,
        # sharey=True, 
        # gridspec_kw={"width_ratios": [.85, 0.15]}
    )

    # Plot the main plot
    sns.scatterplot(
        ax=ax,
        data=data,
        x=query_col,
        y=term_label_col,
        hue=hue_col,
        hue_norm=hue_norm,
        size=size_col,
        sizes=sizes,
        size_norm=size_norm,
        palette=hue_pal,
        markers=markers,
        marker=marker,
        style=style,
        alpha=marker_alpha,
        linewidth=marker_linewidth,
        edgecolor=marker_edgecolor      
    )
    # Additional Styling of the Plot
    # Add grid to x and y
    ax.grid(
        axis=grid_direction,
        color=grid_color,
        alpha=grid_alpha,
        linestyle=grid_linestyle,
        linewidth=grid_linewidth
    )
    # X, Y, and Title
    ax.set_xlabel(
        xlabel,
        fontsize=xlabel_fontsize,
        fontweight=xlabel_fontweight
    )
    ax.set_ylabel(
        ylabel,
        fontsize=ylabel_fontsize,
        fontweight=ylabel_fontweight
    )
    ax.set_title(
        title,
        fontsize = title_fontsize,
        fontweight = title_fontweight,
        x = title_pos[0],
        y = title_pos[1],
        loc = title_loc,
    )
    # X and Y ticks
    ax.tick_params(
        axis="x",
        which="both",
        labelsize=xticklabel_fontsize,
        rotation=xticklabel_rotation,
    )
    ax.tick_params(
        axis="y",
        which="both",
        labelsize=yticklabel_fontsize,
        rotation=yticklabel_rotation,
    )
    # If those are true, flip the x and y ticks
    if xlabel_flip: ax.xaxis.tick_top()
    if xtick_flip: ax.xaxis.set_label_position("top")
    # Legend 
    ax.legend(
        loc="upper center",
        bbox_to_anchor=legend_xypos,
        ncol=legend_ncol,        
        title=legend_title,
        fontsize=legend_fontsize,
        frameon=False,
    )
    # Annotate the plot
    if annotate:
        for i, row in data.iterrows():
            if row[non_annote_col] > min_count:
                if not annotate_non_sig:
                    if not row[marker_col]:
                        continue
                ax.text(
                    x = row[query_col],
                    y = row[term_label_col],
                    s = "{:.1f}".format(row[annotate_col]),
                    ha = "center",
                    va = "center",
                    fontsize = annotate_fontsize,
                    fontweight = annotate_fontweight,
                    color = annotate_lightColor if row[hue_col] > color_threshold else annotate_darkColor
                )

    # Plot margins and tight layout
    ax.margins(
        x=plot_margins[0], 
        y=plot_margins[1]
    )
    if apply_tight_layout:
        fig.tight_layout()
    sns.despine(
        ax=ax,
        top=True,
        right=True,
        left=True,
        bottom=True,
    )

    # Save the figure
    if save:
        save_figures(
            plt.gcf(),
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close()

def enrich_scatterplot(
        data: pd.DataFrame,
        # Data parameters
        x_col: str = "-log10(p_value)",
        term_col: str = "term",
        term_label_col: str = None,
        hue_col: str = "query",
        size_col: str = "GeneRatio",
        marker_col: str = None,
        # Plot parameters
        figsize: tuple = (4, 6),
        hue_pal: dict = None,
        sizes: tuple = (10, 100),
        size_norm: tuple = (0, 1),
        apply_tight_layout: bool = True,
        annotation_threshold: float = None,
        annotation_label: str = "significance cutoff",
        # Style parameters
        markers: list = None, 
        marker_alpha: float = 0.95,
        marker_linewidth: float = 0.25,
        marker_edgecolor: str = "white",
        grid_direction: str = "y", # "x", "y", "both"
        grid_color: str = "grey",
        grid_alpha: float = 0.25,
        grid_linestyle: str = "-",
        grid_linewidth: float = 1.,
        xlabel: str = None,
        xlabel_fontsize: int = 8,
        xlabel_fontweight: str = "normal",
        ylabel: str = None,
        ylabel_fontsize: int = 8,
        ylabel_fontweight: str = "normal",
        title: str = None,
        title_fontsize: int = 10,
        title_fontweight: str = "bold",
        title_pos: tuple = (0.0, 1.025),
        title_loc: str = "center",
        xticklabel_fontsize: int = 8,
        xticklabel_rotation: int = 0,
        yticklabel_fontsize: int = 8,
        yticklabel_rotation: int = 0,
        legend_xypos: tuple = (1, 1),
        legend_ncol: int = 1,
        legend_title: str = None,
        legend_fontsize: int = 8,
        annotation_fontsize: int = 8,
        annotation_fontweight: str = "bold",
        annotation_rotation: int = 90,
        annotation_yoffset: float = 0.025,
        annotation_xoffset: float = 0.025,
        annotation_color: str = "k",
        annotation_linestyle: str = "--",
        annotation_linewidth: float = 1.5,
        annotation_alpha: float = 0.75,
        # Save parameters
        save: bool = False,
        filename: str = "enrich_scatterplot",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
):
    """
        Visualizes the selected enriched terms in a
            scatterplot with queries as colored points.
    """

    if term_label_col is not None:
        y_col = term_label_col
    else:
        y_col = term_col

    if xlabel is None:
        xlabel = x_col
    

    if hue_pal is None:
        hue_pal = "Set1"
    
    if marker_col is None:
        marker = 'o'
        style = None
        markers = None
    else:
        marker = None
        style = marker_col
        if markers is not None:
            markers = {k: markers[k] for k in data[marker_col].unique()}
        else:
            markers = {
                True: 'o',
                False: '.'
            }

    # Initialize the figure
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=figsize,
        # sharey=True,
        # sharex=True,
    )

    sns.scatterplot(
        ax=ax,
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=hue_pal,
        size=size_col,
        sizes=sizes,
        size_norm=size_norm,
        markers=markers,
        marker=marker,
        style=style,
        alpha=marker_alpha,
        linewidth=marker_linewidth,
        edgecolor=marker_edgecolor
    )

    # Additional Styling of the Plot
    # Add grid to x and y
    ax.grid(
        axis=grid_direction,
        color=grid_color,
        alpha=grid_alpha,
        linestyle=grid_linestyle,
        linewidth=grid_linewidth
    )
    # X, Y, and Title
    ax.set_xlabel(
        xlabel,
        fontsize=xlabel_fontsize,
        fontweight=xlabel_fontweight
    )
    ax.set_ylabel(
        ylabel,
        fontsize=ylabel_fontsize,
        fontweight=ylabel_fontweight
    )
    ax.set_title(
        title,
        fontsize = title_fontsize,
        fontweight = title_fontweight,
        x = title_pos[0],
        y = title_pos[1],
        loc = title_loc,
    )
    # X and Y ticks
    ax.tick_params(
        axis="x",
        which="both",
        labelsize=xticklabel_fontsize,
        rotation=xticklabel_rotation,
    )
    ax.tick_params(
        axis="y",
        which="both",
        labelsize=yticklabel_fontsize,
        rotation=yticklabel_rotation,
    )

    # If annotation_treshold is given add a line and label
    if annotation_threshold is not None:
        # Add the annotation line
        ax.axvline(
            x=annotation_threshold,
            color=annotation_color,
            alpha=annotation_alpha,
            linestyle=annotation_linestyle,
            linewidth=annotation_linewidth,           
        )
        # Add the annotation label
        ax.text(
            x=annotation_threshold + annotation_xoffset,
            y=annotation_yoffset,
            s=annotation_label,
            ha="left",
            va="center",
            rotation=annotation_rotation,
            fontsize=annotation_fontsize,
            fontweight=annotation_fontweight,
            color=annotation_color
        )

    # Legend 
    ax.legend(
        loc="upper center",
        bbox_to_anchor=legend_xypos,
        ncol=legend_ncol,        
        title=legend_title,
        fontsize=legend_fontsize,
        frameon=False,
    )


    if apply_tight_layout:
        fig.tight_layout()
    sns.despine(
        ax=ax,
        top=True,
        right=True,
        left=True,
        bottom=False,
    )

    # Save the figure
    if save:
        save_figures(
            plt.gcf(),
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close()

def enrich_single_term(
        data: pd.DataFrame,
        term: str,
        queries: list[str] = None,
        queries_col: str = "query",
        term_col: str = "native",
        term_name_col: str = "name",
        sort_data: bool = True,
        sort_ascd: bool = False,
    
        pvalue: float = 3,
        pvalue_col: str = "-log10(p_value)",
        pvalue_cap: float = 15,
        pvalue_cap_col: str = "-log10(p_capped)",
        figsize: tuple = (6, 5),
        
        size_col: str = "GeneRatio",
        size_norm: tuple = (0.15, 0.75),
        size_range: tuple = (50, 250),
        color: str = "#1f77b4",
        point_edgecolor: str = "black",
        point_linewidth: float = 0.5,
        point_alpha: float = 0.75,
        title_str: str = None,
        title_pad: float = 5,
        title_loc: str = "left",
        title_fontsize: int = 12,
        title_fontweight: str = "normal",
        x_label: str = None,
        x_label_pad: float = 10,
        x_label_fontsize: int = 10,
        x_label_fontweight: str = "normal",
        y_label: str = None,
        y_label_pad: float = 10,
        y_label_fontsize: int = 10,
        y_label_fontweight: str = "normal",
        legend_title: str = None,
        legend_fontsize: int = 8,
        legend_loc: str = "lower right",
        legend_pos: tuple = (1.15, 0),

        # Save parameters
        save: bool = False,
        filename: str = "enrich_singleTerm_scatter",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
            
    ):
    """
        Checks the enrichment data and plots 
            selected term's p-value and enrichment of the queries.
    """
    # Get the data for plotting
    plot_data, term_name = utils.get_term_from_data(
        gp_res = data,
        spec_term = term,
        queries = queries,
        queries_col = queries_col,
        term_col = term_col,
        term_name_col = term_name_col
    )
    
    # If a plot data is returned
    if plot_data is not None:

        if sort_data:
            plot_data = plot_data.sort_values(
                by = pvalue_col,
                ascending = sort_ascd
            )        

        if title_str is None:
            title_str = f"{term}: {term_name}"
        else:
            title_str = "{}\n{}".format(title_str, term_name)

        # If pvalue_cap is not None, use the capped p-value for x-axis
        if pvalue_cap is not None:
            # add a column with capped p-values if it does not exist
            if pvalue_cap_col not in plot_data.columns:
                plot_data[pvalue_cap_col] = plot_data[pvalue_col].apply(
                    lambda x: pvalue_cap if x > pvalue_cap else x
                )
            x_col = pvalue_cap_col
        else:
            # pvalue_cap = pvalue
            x_col = pvalue_col

        if x_label is None:
            x_label = x_col
        
        if y_label is None:
            y_label = queries_col

        if legend_title is None:
            legend_title = size_col

        # Initialize the figure
        fig, ax = plt.subplots(figsize = figsize)        

        # Plot the Groups
        sns.scatterplot(
            ax = ax,
            data = plot_data,
            x = pvalue_col,
            y = queries_col,
            size = size_col,
            sizes = size_range,
            size_norm = size_norm,
            color = color,
            edgecolor = point_edgecolor,
            linewidth = point_linewidth,
            alpha = point_alpha,
            # legend = False, 
        )

        # Define y-ticks with the tissue group names
        ax.set_yticks(
            range(len(plot_data["query"]))
        )
        ax.set_yticklabels(
            plot_data["query"]
        )

        # Set the title
        ax.set_title(
            label = title_str,
            loc = title_loc,
            pad = title_pad,
            fontdict = {
                "fontsize": title_fontsize,
                "fontweight": title_fontweight
            }
        )

        # Set the x-axis label
        ax.set_xlabel(
            xlabel = x_label,
            labelpad = x_label_pad,
            fontdict = {
                "fontsize": x_label_fontsize,
                "fontweight": x_label_fontweight
            }
        )

        # Set the y-axis label
        ax.set_ylabel(
            ylabel = y_label,
            labelpad = y_label_pad,
            fontdict = {
                "fontsize": y_label_fontsize,
                "fontweight": y_label_fontweight
            }
        )

        # Add y gridlines
        ax.grid(
            axis="both",
            color="grey",
            linestyle="-",
            alpha=0.25,
            linewidth=1,
        )

        # Add horizontal line at p-value cutoff
        ax.axvline(
            x=pvalue,
            color="k",
            linestyle="--",
            alpha=0.75, 
            linewidth=1.5
        )
        # Add the annotation rotated 90 degrees to the right
        ax.text(
            x=pvalue + 0.025,
            y=0.025,
            s="p-value cutoff",
            ha="left",
            va="bottom",
            rotation=0,
            fontsize=8,
            fontweight="bold",
            color="k"
        )

        if pvalue_cap is not None:
            # Add vertical line at p-value cutoff
            ax.axvline(
                x=pvalue_cap,
                color="k",
                linestyle="--",
                alpha=0.75, 
                linewidth=1.5
            )
            # Add the annotation rotated 90 degrees to the right
            ax.text(
                x=pvalue_cap + 0.025,
                y=0.025,
                s="Capped p-value",
                ha="left",
                va="bottom",
                rotation=0,
                fontsize=8,
                fontweight="bold",
                color="k"
            )

        ax.legend(
            title = legend_title,
            fontsize = legend_fontsize,
            loc = legend_loc,
            bbox_to_anchor = legend_pos,
            frameon = False
        )

        sns.despine(
            ax = ax,
            top = True,
            right = True,
            left = True,
            bottom = False,
        )

        # Save the figure
        if save:
            save_figures(
                plt.gcf(),
                filename=filename,
                filepath=filepath,
                fileformat=fileformat
            )
            if dont_show:
                plt.close()

    else:
        print("No data to plot.")
        return None
    
def enrich_singleTerm_variability_directions(
        data: pd.DataFrame,
        term: str,
        query: str,
        group: str = "Adult vs Child",

        directions: list = ["Child", "Adult"],
        color_pal: list = ["#1f77b4", "#d62728"],
        vertical: bool = True,
        figsize: tuple = (6, 3),
        
        bar_edgecolor: str = "black",
        bar_linewidth: float = 0.5,
        bar_width: float = 0.8,
        title: str = None,
        title_loc: str = "left",
        title_fontsize: int = 12,
        title_fontweight: str = "normal",
        title_pad: float = 10,
        title_yoffset: float = 1.05,

        legend: bool = True,
        legend_loc: str = "upper center",
        legend_bbox_to_anchor: tuple = (0.5, 1.15),
        legend_ncol: int = 2,
        legend_frameon: bool = False,
        legend_fontsize: float = 10,
        legend_title: str = None,

        line: bool = True,
        line_color: str = "black",
        line_linewidth: float = 1,
        line_linestyle: str = "dashed",
        line_alpha: float = 1,

        # Save parameters
        save: bool = False,
        filename: str = "enrich_singleTerm_variability_direction",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
            
    ):
    """
    
    """

    # If data is None, return None
    if data is None:
        print("No data to plot.")
        return None

    # Check if at least one of the directions are in the UpRegulation column
    if not any([x in directions for x in data["UpRegulation"].unique()]):
        raise ValueError(
            f"""None of the directions {directions} are in the UpRegulation column.
            Please check the directions you provided."""
        )
    

    # Count the Upregulation Directions
    plot_data = data.groupby(
        ["Acc_id", "UpRegulation"]
    ).size().reset_index().pivot_table(
        index="Acc_id",
        columns="UpRegulation",
        values=0
    ).reindex(columns=directions).fillna(0)

    # Assign first direction as negative to show the direction of the change
    plot_data[directions[0]] = -plot_data[directions[0]]

    # Get the number of total pairs 
    npairs = data["Npairs"][0]
    lim = (-npairs, npairs)

    # Initialize the figure
    fig, ax = plt.subplots(
        figsize=figsize,
    )
    
    if vertical:
        kind = "barh"
        xlab = "Number of Pairs (Directional)"
        ylab = "Protein"
        
    else:
        kind = "bar"        
        xlab = "Protein"
        ylab = "Number of Pairs (Directional)"

    # Plot the data
    plot_data.plot(
        kind=kind,
        stacked=True,
        ax=ax,
        color={
            directions[0]: color_pal[0],
            directions[1]: color_pal[1]
        }, 
        edgecolor=bar_edgecolor,
        linewidth=bar_linewidth,
        width=bar_width,
    )

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if vertical:
        ax.set_xlim(lim)
        ax.grid(
            axis="x",
            color="grey",
            linestyle="-",
            alpha=0.25,
            linewidth=1,
        )
        if line:
            ax.axvline(
                x=0,
                color=line_color,
                linewidth=line_linewidth,
                linestyle=line_linestyle,
                alpha=line_alpha,
            )

    else:
        ax.set_ylim(lim)
        ax.grid(
            axis="y",
            color="grey",
            linestyle="-",
            alpha=0.25,
            linewidth=1,
        )
        if line:
            ax.axhline(
                y=0,
                color=line_color,
                linewidth=line_linewidth,
                linestyle=line_linestyle,
                alpha=line_alpha,
            )
    
    # Legend
    if legend:
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=legend_ncol,
            frameon=legend_frameon,
            fontsize=legend_fontsize,
            title=legend_title
        )

    # Build a custom Title
    if title is None:
        title = (
            f"Upregulation Direction in {group} | {query} Comparison" + 
            f"\nTerm = {term}"
        )
    else:
        title = "{}\n{}".format(title, term)

    ax.set_title(
        label = title,
        loc = title_loc,
        fontsize = title_fontsize,
        fontweight = title_fontweight,
        pad = title_pad,
        y = title_yoffset
        
    )

 

    plt.tight_layout()
    sns.despine(
        ax=ax,
        top=True,
        right=True,
        left=True,
        bottom=False,
    )

    # Save the figure
    if save:
        save_figures(
            plt.gcf(),
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close()

def scatterplot_annotated_comparison(
        data: pd.DataFrame,
        s1_cols: list,
        s2_cols: list,
        baseline: float = None, 
        metric_res: dict = None,
        pThr: float = 0.05,
        eqThr: float = 0.75,

        # Figure settings
        figsize: tuple = (4, 4),
        
        # Labels and titles
        xlabel: str = "S1 (log2)",
        ylabel: str = "S2 (log2)",
        title: str = None,
        title_fontsize: int = 12,
        title_loc: str = "left",
        title_pad: float = 10,
        label_fontsize: int = 10,

        # Annotation settings        
        annot_xpos = 0.05,
        annot_ypos = .95,
        annot_fontsize: int = 12,
        annot_color: str = "#2b2d42",
        annot_weight: str = "normal",
        annot_alpha: float = 1,

        ## Main scatterplot Styling
        point_color: str = "#2b2d42",
        point_size: float = 25,
        point_alpha: float = 0.5,
        point_edgecolor: str = "white",
        point_linewidth: float = 0.5,
        point_marker: str = "o",
        point_rasterized: bool = True,

        ## Unity line Styling
        unity_line_color: str = "#d1d1d1",
        unity_line_linewidth: float = 1,
        unity_line_linestyle: str = "--",
        unity_line_alpha: float = 1,

        ## Gridline Styling
        gridline_direction: str = "both",
        gridline_color: str = "lightgrey",
        gridline_linewidth: float = 0.5,
        gridline_linestyle: str = "--",
        gridline_alpha: float = 0.75,

        # Save parameters
        save: bool = False,
        filename: str = "annotated_comparison",
        filepath: str = "",
        fileformat: list[str] = ["png", "svg", "pdf"],
        dont_show: bool = False,
    ):


    # Average replicates for each sample
    s1_avg = data[s1_cols].mean(axis=1)
    s2_avg = data[s2_cols].mean(axis=1)

    # Find the minimum and maximum values
    minVal = np.floor(min(s1_avg.min(), s2_avg.min()))
    maxVal = np.ceil(max(s1_avg.max(), s2_avg.max()))

    # If metric results not provided, calculate them
    if metric_res is None:
        metric_res = utils.collect_metrics(
            data=data,
            s1_cols=s1_cols,
            s2_cols=s2_cols,
            pThr=pThr,
            eqThr=eqThr,
        )
    if baseline is None:
        baseline = "N/A"

    annot_str = """
    Pearson (r) = {pearson:.4f}
    Spearman (p) = {spearman:.4f}
    Kendall (tau) = {kendall:.4f}
    SEI = {equivalence:.4f}
    Obs. Ground Truth = {baseline:.2f}
    """.format(
        pearson = metric_res["pearson (r)"],
        spearman = metric_res["spearman (p)"],
        kendall = metric_res["kendall (tau)"],
        equivalence = metric_res["SEI"],
        baseline = baseline,
    )

    # Initialize the figure 
    fig, ax = plt.subplots(
        figsize=figsize
    )

    # Plot the average values as a scatterplot
    sns.scatterplot(
        ax = ax,
        x = s1_avg,
        y = s2_avg,
        color = point_color,
        s = point_size,
        alpha = point_alpha,
        edgecolor = point_edgecolor,
        linewidth = point_linewidth,
        marker = point_marker,
        rasterized = point_rasterized,
    )

    # Create a unity line
    ax.plot(
        [minVal, maxVal],
        [minVal, maxVal],
        color = unity_line_color,
        linewidth = unity_line_linewidth,
        linestyle = unity_line_linestyle,
        alpha = unity_line_alpha,
    )

    # Add gridlines
    ax.grid(
        axis = gridline_direction,
        color = gridline_color,
        linewidth = gridline_linewidth,
        linestyle = gridline_linestyle,
        alpha = gridline_alpha,
    )

    # Labels
    ax.set_xlabel(
        xlabel,
        fontsize = label_fontsize,
    )
    ax.set_ylabel(
        ylabel,
        fontsize = label_fontsize,
    )
    ax.set_title(
        title,
        fontsize = title_fontsize,
        loc = title_loc,
        pad = title_pad,
    )

    # Add annotations
    ax.text(
        annot_xpos,
        annot_ypos,
        annot_str,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize = annot_fontsize,
        color = annot_color,
        weight = annot_weight,
        alpha = annot_alpha,
    )

    sns.despine(
        ax = ax,
        left = True,
        bottom = True,
    )

    plt.tight_layout()

        # Save the figure
    if save:
        save_figures(
            plt.gcf(),
            filename=filename,
            filepath=filepath,
            fileformat=fileformat
        )
        if dont_show:
            plt.close()