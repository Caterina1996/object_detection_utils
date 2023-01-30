import glob
import pandas as pd

results_csvs=glob.glob("/mnt/c/Users/haddo/yolov5/projects/halimeda/k-fold_trainings/**/**/**results_pascalvoc.csv")
# results_csvs=glob.glob("/mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/**/**/results/**/**results.csv")
max=0
for i,csv in enumerate(results_csvs):
    file_id=csv.split("/") [-4]+"_"+csv.split("/") [-3]
    if "nms" in csv:
        file_id+"_nms"
    print(file_id)

    tmp_df=pd.read_csv(csv)
    tmp_df["train"]=file_id
    print(tmp_df.head())
    
    if i==0:
        # init_df=pd.read_csv(csv)
        # init_df["train"]=file_id
        # print(init_df.head())
        final_df= tmp_df
        
    else:      
        #final_df.append(tmp_df)
        final_df=pd.concat([final_df, tmp_df])
        print(final_df.head())


print(final_df.head())
#print(final_df.describe())

final_df.to_csv("/mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/test_metrics.csv")