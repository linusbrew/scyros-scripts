import polars as pl
import numpy as np
import seaborn as sns
import matplotlib as plt
import os
from scripts.StatisticsExtractor import StatisticsExtractor, PREFIX

PATH_KEYWORDS = "keywords/"

df_files = pl.read_csv("result/" + PREFIX + "_dedup_files.csv")

file_name = "stats/stats_files.txt"
os.makedirs(os.path.dirname(file_name), exist_ok=True)

extractor = StatisticsExtractor()

# TODO: min() is probably not very useful since it will always be 0
with open(file_name, "w") as f:
    file_names = extractor.file_to_kw.keys()
    import_percent = extractor.percentage_imports(
        df_files, PATH_KEYWORDS + "import.json", PATH_KEYWORDS + "from_import.json"
    )
    f.write(
        f"The percentage of all imports being 'import foo' is {import_percent * 100}%\n"
    )
    f.write(
        f"The percentage of all imports being 'from bar import foo' is {(1 - import_percent) * 100}%\n"
    )
    f.write("\n")
    f.write("\n")
    for kw in file_names:
        path = PATH_KEYWORDS + kw
        f.write(f"----------{extractor.file_to_kw.get(kw)}----------\n")
        # avg_loc = extractor.avg_length(df_files, "loc", path)
        # avg_words = extractor.avg_length(df_files, "words", path)
        # f.write(f"On average when {extractor.file_to_kw.get(kw)} appears the file has: {avg_loc} LOC\n")
        # f.write(f"On average when {extractor.file_to_kw.get(kw)} appears the file has: {avg_words} words\n")

        files_kw_percentage = extractor.kw_percentage(df_files, path)
        # loc_kw_percentage = extractor.kw_ratio_dataframe(df_projects, "loc_of_files_with_" + path, "loc")
        # words_kw_percentage = extractor.kw_ratio_dataframe(df_projects, "words_of_files_with_" + path, "words")
        f.write(
            f"The percentage of files having {extractor.file_to_kw.get(kw)}: {files_kw_percentage}%\n"
        )

        kw_in_file = extractor.kw_in_project(df_files, path)
        f.write(
            f"The total number of occurrences in all files for {extractor.file_to_kw.get(kw)}: {kw_in_file}\n"
        )

        max_kw_file = extractor.max_keyword_project(df_files, path)
        f.write(
            f"The maximum number of occurrences for {extractor.file_to_kw.get(kw)} is: {max_kw_file}\n"
        )

        min_kw_file = extractor.min_keyword_project(df_files, path)
        f.write(
            f"The minimum number of occurrences for {extractor.file_to_kw.get(kw)} is: {min_kw_file}\n"
        )

        mean_kw_file = extractor.calculate_mean(df_files, path)
        f.write(
            f"The mean of keywords is for {extractor.file_to_kw.get(kw)}: {mean_kw_file}\n"
        )

        median_kw_file = extractor.calculate_median(df_files, path)
        f.write(
            f"The median of keywords is for {extractor.file_to_kw.get(kw)}: {median_kw_file}\n"
        )

        var_kw_file = extractor.calculate_variance(df_files, path)
        f.write(
            f"The variance of keywords for {extractor.file_to_kw.get(kw)} is : {var_kw_file}\n"
        )

        sigma_kw_file = extractor.calculate_sigma(df_files, path)
        f.write(
            f"The standard deviation of keywords for {extractor.file_to_kw.get(kw)} is : {sigma_kw_file}\n"
        )

        quantiles_kw_file = extractor.calculate_quant(df_files, path)
        f.write(
            f"The first quartile of keywords for {extractor.file_to_kw.get(kw)} is : {quantiles_kw_file[0]}\n"
        )
        f.write(
            f"The third quartile of keywords for {extractor.file_to_kw.get(kw)} is : {quantiles_kw_file[2]}\n"
        )
        f.write(
            f"The interquartile range of keywords for {extractor.file_to_kw.get(kw)} is : {quantiles_kw_file[2] - quantiles_kw_file[0]}\n"
        )
        f.write("\n")
        f.write("\n")
