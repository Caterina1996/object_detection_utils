
results_csvs=glob.glob("/mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/**/inference_test_2/coverage2/metrics/*.xlsx")
max=0

print("CSVS:",results_csvs)

for i,csv in enumerate(results_csvs):
    file_id=csv.split("/") [-1]
    root=csv.split("/results")[0]
    columns_name_list = list()

    path_file = os.path.join(root, file)
    excel = pd.ExcelFile(path_file, engine='openpyxl')
    sheets = excel.sheet_names
    data = excel.parse(sheets[0])

    for i in range(data.shape[1]):
        columns_name_list.append(str(data.columns[i]))

    data = pd.DataFrame(data, columns=columns_name_list)

    data_np = data.to_numpy()

    name = os.path.join(root, file[1])

    rows_name_list.append(name)
    for i in range(data_np.shape[0]-1):
        rows_name_list.append(" ")
    rows_data_list.append(data_np)

    rows_data = np.vstack(rows_data_list)
    df = pd.DataFrame(data=rows_data, index=rows_name_list, columns=columns_name_list)
    filepath = os.path.join(path_out, lookfor + "thr_unified.xlsx")
    df.to_excel(filepath, index=True)