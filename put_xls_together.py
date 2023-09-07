import glob
import os
import pandas as pd
import numpy as np

def main():

    results_csvs = glob.glob("/mnt/c/Users/haddo/Halimeda/merge/test/merged_weights_yolo_sizes/**/weighted_merge/metrics/*.xlsx")
    # Example: /mnt/c/Users/haddo/Halimeda/merge/test/merged_weights_yolo_sizes/yolo_XL/weighted_merge/metrics/metrics.xlsx

    rows_data_list = []
    rows_name_list = []

    print("CSVS:", results_csvs)

    for i, csv in enumerate(results_csvs):

        columns_name_list = list()

        excel = pd.ExcelFile(csv, engine='openpyxl')
        sheets = excel.sheet_names
        data = excel.parse(sheets[0])

        for i in range(data.shape[1]):
            columns_name_list.append(str(data.columns[i]))

        data = pd.DataFrame(data, columns=columns_name_list)

        data_np = data.to_numpy()

    
        rows_name_list.append(csv)
        
        for i in range(data_np.shape[0]-1):
            rows_name_list.append(None)
        rows_name_list.append('')

        rows_data_list.append(data_np)

    rows_data = np.vstack(rows_data_list)
    df = pd.DataFrame(data=rows_data, index=rows_name_list, columns=columns_name_list)

    path_out = "/path/to/output/directory"
    lookfor = "filename_prefix_"
    filepath = os.path.join(path_out, lookfor + "thr_unified.xlsx")
    df.to_excel(filepath, index=True)


if __name__ == '__main__':
    main()


#-----------------------------------------

