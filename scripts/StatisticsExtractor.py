import polars as pl
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

# Absolute path to the keywords directory
paths = "/home/linus-brewitz/Code/thesis/scyros/keywords"

# Prefix for files generated from Scyros
PREFIX = "py"


class StatisticsExtractor:
    """
    A class that calculates a variety of different statistics about
    Polars dataframes.

    Attributes
    ----------
    quantiles : list
        A list of the percentiles for the first, second, and third quartiles
    files_tp_kw : dict
        A dictionary mapping files to their corresponding keyword
    PATH_KEYWORDS : str
        Constant for file path to keywords
    PATH_FIGURES : str
        Constant for file path to figures
    THRESHOLD : int
        Constant for how many files Fred is willing to go through.
    """

    quartiles = [0.25, 0.50, 0.75]
    file_to_kw = dict()
    PATH_KEYWORDS = "keywords/"
    PATH_FIGURES = "figures/"
    THRESHOLD = 200

    # NOTE: this is assuming there is only ONE keyword per file, this is also extremely
    # hardcoded.
    def __init__(self):
        """The method that initiliases the class by loading the dictionary
        with file names from the keywords."""
        for e in os.scandir(paths):
            if e.is_file():
                with open(e.path, "r") as f:
                    data = json.load(f)
                    new_string = data["keywords"][0].replace("\\t", "")
                    new_string = new_string.replace("\\r", "")
                    new_string = new_string.replace("\\v", "")
                    new_string = new_string.replace("\\S", "")
                    new_string = new_string.replace("\\s", "")
                    new_string = new_string.replace("\\b", "")
                    new_string = new_string.replace("\\f", "")
                    new_string = new_string.replace("\\n", "")
                    new_string = re.sub(r"[^A-Za-z\.]", "", new_string)
                    if new_string[0] == "m":
                        new_string = new_string.replace("m", "", 1)
                    self.file_to_kw[e.name] = new_string

    def below_threshold(self, df: pl.DataFrame, kw: str) -> bool:
        """Checks whether a certain keyword in a project is below or
        equal to a defined threshold

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        kw : str
            The keyword to search for

        Returns
        -------
        bool
            bool if the keyword occurs less or equal times
            compared to the threshold
        """

        return self.kw_in_project(df, kw) <= self.THRESHOLD

    # I'm thinking this method should not return anything but simply
    # write to a .csv file instead to save screen space.
    def get_project_list(self, df: pl.DataFrame, kw: str) -> None:
        """Gathers files in which a keyword occurs less or equal to a
        defined threshold and writes the gathered files to
        a .csv file

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        kw : str
            The keyword to search for

        Returns
        -------
        None
            method does not return, but instead writes to
            .csv file
        """
        if self.below_threshold(df, kw):
            os.makedirs(
                os.path.dirname("result/filtered_" + kw + ".csv"), exist_ok=True
            )
            new_df = pl.DataFrame({"id": [], "path": [], "name": [], kw: []})
            filtered = df.filter((pl.col(kw) > 0))
            new_df = filtered.select(
                pl.col("id"), pl.col("path"), pl.col("name"), pl.col(kw)
            )
            new_df.write_csv("result/filtered_" + kw + ".csv")
            print(new_df)

    def plot_lorenz(self, arr, kw: str) -> None:
        """Plots a Lorenz curve and saves the figure
        as a PDF

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        kw : str
            The keyword to search for

        Returns
        -------
        None
            method write the figure to a PDF
        """

        gini_coeff = self.gini(arr)
        lorenz_curve = self.lorenz(arr)
        x_line = self.last_zero(arr)

        plt.cla()
        plt.clf()

        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        # we need the X values to be between 0.0 to 1.0
        kw = kw.replace(".json", "")
        ax.plot(
            np.linspace(0.0, 1.0, lorenz_curve.size),
            lorenz_curve,
            label="Lorenz curve for " + kw,
            zorder=2,
        )

        # plot the straight line perfect equality curve
        ax.plot([0, 1], [0, 1], label="Equality line", zorder=2)

        ax.axvline(x_line, color="r", label="Last zero", zorder=2)

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.025,
            0.77,
            "Gini = " + str(gini_coeff),
            bbox=props,
            zorder=3,
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xlabel("Cumulative Share of projects")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_ylabel("Cumulative Share of keywords/functions")
        l = ax.legend(loc="upper left")
        l.set_zorder(3)

        plt.savefig(self.PATH_FIGURES + kw + ".pdf")
        plt.close(fig)

    def gini(self, g_arr) -> float:
        """Calculates the Gini coefficient for an array of values

        This method was originally posted on GitHubGist by CMCDragonkai
        Link: https://gist.github.com/CMCDragonkai/c79b9a0883e31b327c88bfadb8b06fc4

        Parameters
        ----------
        kw : str
            A sorted NumPy array of values

        Returns
        -------
        float
            the Gini coefficient as a float
        """
        count = g_arr.size
        coefficient = 2 / count
        indexes = np.arange(1, count + 1)
        weighted_sum = (indexes * g_arr).sum()
        total = g_arr.sum()
        constant = (count + 1) / count
        return round(coefficient * weighted_sum / total - constant, 4)

    def lorenz(self, l_arr: np._Array1D) -> np._Array1D:
        """Calculates the Lorenz array

        This method was originally posted on GitHubGist by CMCDragonkai
        Link: https://gist.github.com/CMCDragonkai/c79b9a0883e31b327c88bfadb8b06fc4

        Parameters
        ----------
        l_arr : _Array1D
            Array of sorted values

        Returns
        -------
        np._Array1D
            array of values
        """

        # this divides the prefix sum by the total sum
        # this ensures all the values are between 0 and 1.0
        scaled_prefix_sum = l_arr.cumsum() / l_arr.sum()
        # this prepends the 0 value (because 0% of all people have 0% of all wealth)
        return np.insert(scaled_prefix_sum, 0, 0)

    def last_zero(self, arr) -> float:
        """Calculates the last index where a 0 is. Then converts to
        decimal form to get share where the index is

        Parameters
        ----------
        arr : np._Array1D
            Sorted array of values

        Returns
        -------
        float
            the share of the population where the last
            0 occurs
        """
        return (np.where(arr == 0)[0][-1] + 1) / len(arr)

    def plot_stacked_bar_imports(
        self, kw: str, num_module: int, num_function: int
    ) -> None:
        """Plots a stacked bar figure for a keyword and then saves
        the figure as a PDF

        Parameters
        ----------
        kw : str
            The keyword to create the figure for
        num_module : int
            The number of occurrences at the module level
        num_function : int
            The number of occurrences at the function level

        Returns
        -------
        None
            saves the figure as a PDF
        """
        plt.cla()
        plt.clf()
        kw = kw.replace(".json", "")
        imports = [num_module, num_function]
        names = [kw + " - module level", kw + " - function level"]
        colors = ["steelblue", "darkorange"]
        total = sum(imports)

        fig, ax = plt.subplots()
        bars = ax.bar(names, imports, color=colors, edgecolor="white")

        # Show percentage above each bar
        for bar, count in zip(bars, imports):
            pct = count / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel("Count")
        ax.set_title("Import location distribution")
        ax.spines[["top", "right"]].set_visible(False)
        plt.savefig(self.PATH_FIGURES + kw + "_bar.pdf")
        plt.close(fig)

    def count_parse_error(self, df: pl.DataFrame) -> int:
        """Counts the number of parsing errors in a
        given Polars dataframe

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in

        Returns
        -------
        int
            the number of parsing errors that has occurred
        """
        filtered = df.filter(pl.col("parse_error") != "none")
        result = filtered.select(pl.col("parse_error").count()).item()
        return result

    def write_parse_error(self, df: pl.DataFrame, path: str) -> None:
        """Writes all of the files that has a parsing error
        to a .csv file

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        path : str
            The path to either a function (func) or logs (logs) file

        Returns
        -------
        None
            writes to a .csv file
        """
        os.makedirs(os.path.dirname(f"result/parse_error_{path}.csv"), exist_ok=True)
        filtered = df.filter(pl.col("parse_error") != "none")
        filtered.write_csv(f"result/parse_error_{path}.csv")

    def column_as_numpy(self, df: pl.DataFrame, column: str) -> np._Array1D:
        """Takes a dataframe and a column name and then returns that
        column as a NumPy array

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to use
        column : str
            The column name to search for

        Returns
        -------
        _Array1D
            the column specified as a NumPy array
        """
        sorted = df.sort(column)
        selected_col = sorted.select(pl.col(column))
        result = selected_col.to_numpy()
        return np.ravel(result)

    def count_rows(self, df: pl.DataFrame) -> int:
        """Counts how many rows there are in a Polars dataframe

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in

        Returns
        -------
        int
            how many rows there are in the dataframe
        """
        return df.select(pl.count()).item()

    def num_after_cleanup(self, df_before: pl.DataFrame, df_after: pl.DataFrame) -> int:
        """Counts the number of rows removed when cleaning up the df_before
        dataframe

        Parameters
        ----------
        df_before : pl.DataFrame
            The dataframe before cleaning it up
        df_after : pl.DataFrame
            The dataframe after cleaning it up

        Returns
        -------
        int
            the number of rows removed from the dataframe
        """
        return self.count_rows(df_before) - self.count_rows(df_after)

    def percentage_after_cleanup(
        self, df_before: pl.DataFrame, df_after: pl.DataFrame
    ) -> float:
        """Checks whether a certain keyword in a project is below or
        equal to a defined threshold

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        kw : str
            The keyword to search for

        Returns
        -------
        bool
            bool if the keyword occurs less or equal times
            compared to the threshold
        """
        return round((self.count_rows(df_after) / self.count_rows(df_before)) * 100, 2)

    def percentage_imports(self, df: pl.DataFrame, numerator: str, deno: str) -> float:
        """Calculates the percentage of imports being a certain type.

        The two variants are "import foo" and "from foo import bar"

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        numerator : str
            One of two import variants
        denominator : str
            One of two import variants

        Returns
        -------
        bool
            bool if the keyword occurs less or equal times
            compared to the threshold
        """
        numerator_num = df.select(pl.col(numerator).sum().round(3)).item()
        deno_num = df.select(pl.col(deno).sum().round(3)).item()
        percentage = numerator_num / (numerator_num + deno_num)
        return round(percentage, 3)

    def calculate_share_functions_with_keyword(
        self, df: pl.DataFrame, kw: str, column: str
    ) -> float:
        """Function that calculates the share of a certain keyword
        compared to all functions

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        kw : str
            The keyword to search for
        column : str
            The column to compare the keyword column to

        Returns
        -------
        float
            the share of functions that have keywords
        """
        return df.select(
            ((pl.col(kw).sum() / pl.col(column).sum()) * 100).round(2)
        ).item()

    def avg_length(self, df: pl.DataFrame, length: str, keyword: str) -> float:
        """Calculates the average length of a file if it contains
        a keyword. The length can be either LOC or words

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        length : str
            The variant of length to search for (LOC/words)
        keyword : str
            The keyword to search for

        Returns
        -------
        float
            the average length of a file
        """
        non_zero = df.remove(pl.col(keyword) == 0)
        return non_zero.select(pl.col(length).mean().round(2)).item()

    def median_length(self, df: pl.DataFrame, length: str, keyword: str) -> int:
        """Calculates the median length of a file if it contains
        a keyword. The length can be either LOC or words

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        length : str
            The variant of length to search for (LOC/words)
        keyword : str
            The keyword to search for

        Returns
        -------
        int
            the median length of a file
        """
        non_zero = df.remove(pl.col(keyword) == 0)
        return non_zero.select(pl.col(length).median().round(2)).item()

    # NOTE: just in case I need to clean the data because of the errors
    def clean_projects(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cleans a project dataframe from errors if a row
        in the "path" column has "error" as value by removing
        those rows

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in

        Returns
        -------
        pl.DataFrame
            the dataframe without rows containing "error"
        """
        return df.remove(pl.col("path") == "error")

    def kw_in_project(self, df: pl.DataFrame, column: str) -> int:
        """Counts how many occurrences there are of a certain
        keyword

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        column : str
            The column to sum over

        Returns
        -------
        int
            the number of occurrence of a keyword
        """
        kw_count = df.select(pl.col(column)).sum().item()
        return kw_count

    def kw_percentage(self, df: pl.DataFrame, column: str) -> float:
        """Calculates the percentage of projects/files/functions
        having at least one occurrence of a column


        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        column : str
            The column to search for

        Returns
        -------
        float
            the percentage of projects/files/functions having at least
            one occurrence of the specified column
        """
        df_filtered = df.filter(pl.col(column) > 0)
        result = df_filtered.select(pl.count()).item()
        return round((result / df.select(pl.count()).item()) * 100, 2)

    def get_kw_ratio(self, df: pl.DataFrame, kw: str) -> float:
        """Calculates the percentage of projects/files/functions
        having at least one occurrence of a keyword

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        kw : str
            The keyword to search for

        Returns
        -------
        float
            the percentage of projects/files/functions having at least
            one occurrence of the specified column
        """
        df_filtered = df.filter(pl.col(kw) > 0)
        result = df_filtered.select(pl.count()).item()
        length_original = df.select(pl.count()).item()
        return round((result / length_original) * 100, 2)

    # To see how the ratio between the total LOC and the LOC that contain
    # the keyword. This is only intended if we have one keyword, must
    # be updated when doing this with multiple keywords
    # NOTE: this actually works for words as well since we specify
    # the columns before the call
    def kw_ratio_dataframe(self, df: pl.DataFrame, part: str, total: str) -> float:
        """Calculates the ratio of projects/files/functions for
        LOC/words/functions that have keywords.


        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        path : str
            The column the is documented as "foo_with_bar"
        total: str
            The column that symbolises the total column

        Returns
        -------
        bool
            bool if the keyword occurs less or equal times
            compared to the threshold

        Example(s)
        ----------
        >>> ratio = kw_ratio_dataframe(df_projects, "files_with_kw", "files")
        >>> print(ratio)
        42.37
        """
        return df.select(
            ((pl.col(part).sum() / pl.col(total).sum()) * 100).round(2)
        ).item()

    def max_keyword_project(self, df: pl.DataFrame, column: str) -> int:
        """Calculates the maximum value for a column in a
        Polars dataframe

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        column : str
            The column to search for

        Returns
        -------
        int
            the maximum value in the specified column
        """
        return df.select(pl.col(column).max()).item()

    def min_keyword_project(self, df: pl.DataFrame, column: str) -> int:
        """Calculates the minimum value for a column in a
        Polars dataframe

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        column : str
            The column to search for

        Returns
        -------
        int
            the minimum value in the specified column
        """
        return df.select(pl.col(column).min()).item()

    def calculate_mean(self, df: pl.DataFrame, column: str) -> float:
        """Calculates the mean value for a column in a
        Polars dataframe

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        column : str
            The column to search for

        Returns
        -------
        float
            the mean value in the specified column
        """
        df_mean = df.mean()
        return df_mean.select(pl.col(column).round(2)).item()

    def calculate_median(self, df: pl.DataFrame, column: str) -> int:
        """Calculates the median value for a column in a
        Polars dataframe

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        column : str
            The column to search for

        Returns
        -------
        int
            the median value in the specified column
        """
        df_median = df.median()
        return df_median.select(pl.col(column)).item()

    def calculate_variance(self, df: pl.DataFrame, column: str) -> float:
        """Calculates the variance value for a column in a
        Polars dataframe

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        column : str
            The column to search for

        Returns
        -------
        float
            the variance value in the specified column
        """
        df_var = df.var()
        return round(df_var.select(pl.col(column)).item(), 2)

    def calculate_sigma(self, df: pl.DataFrame, column: str) -> float:
        """Calculates the standard deviation for a column in a
        Polars dataframe

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        column : str
            The column to search for

        Returns
        -------
        float
            the standard deviation value in the specified column
        """
        df_sigma = df.std()
        return round(df_sigma.select(pl.col(column)).item(), 2)

    def calculate_quant(self, df: pl.DataFrame, column: str) -> tuple[int, int, int]:
        """Calculates the first, second, and third quantiles for a column in a
        Polars dataframe

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to look in
        column : str
            The column to search for

        Returns
        -------
        tuple
            a tuple with tree elements, the first value corresponds
            the first quartile, seconds value corresponds to
            second quartile (median), third value corresponds to
            third quartile
        """
        result = list()
        df_col = df.select(pl.col(column))
        for quantile in self.quartiles:
            result.append(df_col.quantile(quantile=quantile).item())
        return tuple(result)
