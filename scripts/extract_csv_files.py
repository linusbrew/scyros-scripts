from StatisticsExtractor import StatisticsExtractor, PREFIX
import polars as pl

PATH_KEYWORDS = "keywords/"
df_project = pl.read_csv("result/" + PREFIX + "_projects_clean.csv")
df_files = pl.read_csv("result/" + PREFIX + "_dedup_files.csv")
df_logs = pl.read_csv("result/" + PREFIX + "_function_logs.csv")
df_functions = pl.read_csv("result/" + PREFIX + "_functions.csv")

extractor = StatisticsExtractor()
file_names = extractor.file_to_kw.keys()

extractor.write_parse_error(df_logs, "logs")
extractor.write_parse_error(df_functions, "func")

for kw in file_names:
    path = PATH_KEYWORDS + kw
    extractor.get_project_list(df_project, path)
