import polars as pl
import numpy as np
import seaborn as sns
import matplotlib as plt
import os
from scripts.StatisticsExtractor import StatisticsExtractor, PREFIX

PATH_KEYWORDS = "keywords/"

df_functions = pl.read_csv("result/" + PREFIX + "_functions.csv")
df_logs = pl.read_csv("result/" + PREFIX + "_function_logs.csv")

file_name = "stats/stats_functions.txt"
os.makedirs(os.path.dirname(file_name), exist_ok=True)

extractor = StatisticsExtractor()

# TODO: min() is probably not very useful since it will always be 0
with open(file_name, "w") as f:
    functions_w_kw = extractor.calculate_share_functions_with_keyword(
        df_logs, "functions_with_kw", "functions"
    )
    f.write(f"{functions_w_kw}% of all functions have at least one keyword")
    f.write("\n\n")

    file_names = extractor.file_to_kw.keys()
    for kw in file_names:
        path = PATH_KEYWORDS + kw
        f.write(f"----------{extractor.file_to_kw.get(kw)}----------\n")
        share_all_func = extractor.calculate_share_functions_with_keyword(
            df_logs, path, "functions"
        )
        share_kw_func = extractor.calculate_share_functions_with_keyword(
            df_logs, path, "functions_with_kw"
        )

        f.write(
            f"{extractor.file_to_kw.get(kw)} appears in {share_all_func}% of all functions\n"
        )
        f.write(
            f"{extractor.file_to_kw.get(kw)} appears in {share_kw_func}% of functions with keywords\n"
        )

        avg_loc = extractor.avg_length(df_functions, "loc", path)
        avg_words = extractor.avg_length(df_functions, "words", path)
        f.write(
            f"On average {extractor.file_to_kw.get(kw)} appears in a function with : {avg_loc} LOC\n"
        )
        f.write(
            f"On average {extractor.file_to_kw.get(kw)} appears in a function with : {avg_words} words\n"
        )

        median_loc = extractor.median_length(df_functions, "loc", path)
        median_words = extractor.median_length(df_functions, "words", path)
        f.write(
            f"The median value when {extractor.file_to_kw.get(kw)} appears in a function is : {median_loc} LOC\n"
        )
        f.write(
            f"The median value when {extractor.file_to_kw.get(kw)} appears in a function is : {median_words} words\n"
        )

        kw_in_function = extractor.kw_in_project(df_functions, path)
        f.write(
            f"The total number of occurrences in all functions for {extractor.file_to_kw.get(kw)}: {kw_in_function}\n"
        )

        max_kw_function = extractor.max_keyword_project(df_functions, path)
        f.write(
            f"The maximum number of occurrences for {extractor.file_to_kw.get(kw)} is: {max_kw_function}\n"
        )

        min_kw_function = extractor.min_keyword_project(df_functions, path)
        f.write(
            f"The minimum number of occurrences for {extractor.file_to_kw.get(kw)} is: {min_kw_function}\n"
        )

        mean_kw_function = extractor.calculate_mean(df_functions, path)
        f.write(
            f"The mean of keywords is for {extractor.file_to_kw.get(kw)}: {mean_kw_function}\n"
        )

        median_kw_function = extractor.calculate_median(df_functions, path)
        f.write(
            f"The median of keywords is for {extractor.file_to_kw.get(kw)}: {median_kw_function}\n"
        )

        var_kw_function = extractor.calculate_variance(df_functions, path)
        f.write(
            f"The variance of keywords for {extractor.file_to_kw.get(kw)} is : {var_kw_function}\n"
        )

        sigma_kw_function = extractor.calculate_sigma(df_functions, path)
        f.write(
            f"The standard deviation of keywords for {extractor.file_to_kw.get(kw)} is : {sigma_kw_function}\n"
        )

        quantiles_kw_function = extractor.calculate_quant(df_functions, path)
        f.write(
            f"The first quartile of keywords for {extractor.file_to_kw.get(kw)} is : {quantiles_kw_function[0]}\n"
        )
        f.write(
            f"The third quartile of keywords for {extractor.file_to_kw.get(kw)} is : {quantiles_kw_function[2]}\n"
        )
        f.write(
            f"The interquartile range of keywords for {extractor.file_to_kw.get(kw)} is : {quantiles_kw_function[2] - quantiles_kw_function[0]}\n"
        )
        f.write("\n")
        f.write("\n")
