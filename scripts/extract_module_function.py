from scripts.StatisticsExtractor import StatisticsExtractor, PREFIX
import polars as pl
import os


PATH_KEYWORDS = "keywords/"

df_files = pl.read_csv("result/" + PREFIX + "_dedup_files.csv")
df_logs = pl.read_csv("result/" + PREFIX + "_function_logs.csv")

file_name = "stats/stats_imports.txt"
os.makedirs(os.path.dirname(file_name), exist_ok=True)

extractor = StatisticsExtractor()
with open(file_name, "w") as f:
    file_names = extractor.file_to_kw.keys()
    for kw in file_names:
        path = PATH_KEYWORDS + kw
        kw_in_function = extractor.kw_in_project(df_logs, path)
        kw_in_files = extractor.kw_in_project(df_files, path)
        kw_in_module = kw_in_files - kw_in_function
        f.write(f"----------{extractor.file_to_kw.get(kw)}----------\n")
        f.write(
            f"The amount of {extractor.file_to_kw.get(kw)} at module level: {kw_in_module}\n"
        )
        f.write(
            f"The amount of {extractor.file_to_kw.get(kw)} at function level: {kw_in_function}\n"
        )

        kw_in_module_percentage = kw_in_module / kw_in_files
        kw_function_percentage = kw_in_function / kw_in_files
        f.write(
            f"The percentage of keywords at module level is {round(kw_in_module_percentage * 100, 2)}%\n"
        )
        f.write(
            f"The percentage of keywords at function level is {round(kw_function_percentage * 100, 2)}%\n"
        )

        extractor.plot_stacked_bar_imports(kw, kw_in_module, kw_in_function)

        f.write("\n")
        f.write("\n")
