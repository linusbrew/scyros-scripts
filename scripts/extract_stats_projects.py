import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scripts.StatisticsExtractor import StatisticsExtractor, PREFIX

PATH_KEYWORDS = "keywords/"
PATH_FIGURES = "figures/"


stats_file_name = "stats/stats_projects.txt"
os.makedirs(os.path.dirname(stats_file_name), exist_ok=True)

extractor = StatisticsExtractor()

df_before = pl.read_csv("result/" + PREFIX + "_projects.csv")
df_projects = extractor.clean_projects(df_before)
df_projects.write_csv("result/" + PREFIX + "_projects_clean.csv")

with open(stats_file_name, "w") as f:
    projects_kw_ratio = extractor.get_kw_ratio(df_projects, "files_with_kw")
    f.write(
        f"The percentage of projects having at least one keyword: {projects_kw_ratio}%\n"
    )
    files_kw_percentage = extractor.kw_ratio_dataframe(
        df_projects, "files_with_kw", "files"
    )
    f.write(
        f"The percentage of files having at least one keyword: {files_kw_percentage}%\n"
    )
    f.write(
        f"The number of projects before cleanup: {extractor.count_rows(df_before)}\n"
    )
    f.write(
        f"The number of projects after cleanup: {extractor.count_rows(df_projects)}\n"
    )
    f.write(
        f"The number of projects: {extractor.num_after_cleanup(df_before, df_projects)}, that are no longer reachable\n"
    )
    f.write(
        f"The percentage: {100 - extractor.percentage_after_cleanup(df_before, df_projects)}%, that are no longer reachable\n"
    )
    f.write(
        f"The percentage: {extractor.percentage_after_cleanup(df_before, df_projects)}%, that remain of the dataset\n"
    )
    f.write("\n")
    f.write("\n")

    file_names = extractor.file_to_kw.keys()

    for kw in file_names:
        path = PATH_KEYWORDS + kw
        figure_file_name = PATH_FIGURES + kw + ".jpg"

        os.makedirs(os.path.dirname(figure_file_name), exist_ok=True)
        arr = extractor.column_as_numpy(df_projects, PATH_KEYWORDS + kw)
        extractor.plot_lorenz(arr, kw)

        f.write(f"----------{extractor.file_to_kw.get(kw)}----------\n")
        gini_coeff = extractor.gini(arr)
        f.write(
            f"The Gini coefficient for {extractor.file_to_kw.get(kw)} is {gini_coeff}\n"
        )

        files_kw_percentage = extractor.kw_ratio_dataframe(
            df_projects, "files_with_" + path, "files"
        )
        # loc_kw_percentage = extractor.kw_ratio_dataframe(df_projects, "loc_of_files_with_" + path, "loc")
        # words_kw_percentage = extractor.kw_ratio_dataframe(df_projects, "words_of_files_with_" + path, "words")
        f.write(
            f"The percentage of files having {extractor.file_to_kw.get(kw)}: {files_kw_percentage}%\n"
        )
        # f.write(f"The percentage of LOC having {extractor.file_to_kw.get(kw)}: {loc_kw_percentage}%\n")
        # f.write(f"The percentage of words having {extractor.file_to_kw.get(kw)}: {words_kw_percentage}%\n")

        projects_kw_percentage = extractor.kw_percentage(df_projects, path)
        f.write(
            f"The percentage of projects having {extractor.file_to_kw.get(kw)}: {projects_kw_percentage}%\n"
        )

        kw_in_project = extractor.kw_in_project(df_projects, path)
        f.write(
            f"The total number of occurrences in all projects for {extractor.file_to_kw.get(kw)}: {kw_in_project}\n"
        )

        max_kw_project = extractor.max_keyword_project(df_projects, path)
        f.write(
            f"The maximum number of occurrences for {extractor.file_to_kw.get(kw)} in a project is: {max_kw_project}\n"
        )

        min_kw_project = extractor.min_keyword_project(df_projects, path)
        f.write(
            f"The minimum number of occurrences for {extractor.file_to_kw.get(kw)} in a project is: {min_kw_project}\n"
        )

        mean_kw_project = extractor.calculate_mean(df_projects, path)
        f.write(
            f"The mean of keywords is for {extractor.file_to_kw.get(kw)}: {mean_kw_project}\n"
        )

        median_kw_project = extractor.calculate_median(df_projects, path)
        f.write(
            f"The median of keywords is for {extractor.file_to_kw.get(kw)}: {median_kw_project}\n"
        )

        var_kw_project = extractor.calculate_variance(df_projects, path)
        f.write(
            f"The variance of keywords for {extractor.file_to_kw.get(kw)} is : {var_kw_project}\n"
        )

        sigma_kw_project = extractor.calculate_sigma(df_projects, path)
        f.write(
            f"The standard deviation of keywords for {extractor.file_to_kw.get(kw)} is : {sigma_kw_project}\n"
        )

        quantiles_kw_project = extractor.calculate_quant(df_projects, path)
        f.write(
            f"The first quartile of keywords for {extractor.file_to_kw.get(kw)} is : {quantiles_kw_project[0]}\n"
        )
        f.write(
            f"The third quartile of keywords for {extractor.file_to_kw.get(kw)} is : {quantiles_kw_project[2]}\n"
        )
        f.write(
            f"The interquartile range of keywords for {extractor.file_to_kw.get(kw)} is : {quantiles_kw_project[2] - quantiles_kw_project[0]}\n"
        )
        f.write("\n")
        f.write("\n")
