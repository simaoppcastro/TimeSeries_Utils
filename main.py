import keras
import matplotlib
import numpy as np
import pandas as pd
import urllib3
import tensorflow as tf
import _utils as ut
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

def joinLocalData():
    path1 ='.csv'
    path2 ='.csv'
    path3 ='.csv'
    path4 ='.csv'
    path5 ='.csv'

    label1, data1 = ut.loadLocalData(sPath=path1, bHeader=None)
    # print(data1.shape[0], data1.shape[1])
    label2, data2 = ut.loadLocalData(sPath=path2, bHeader=None)
    # print(data2.shape[0], data2.shape[1])
    label3, data3 = ut.loadLocalData(sPath=path3, bHeader=None)
    # print(data3.shape[0], data3.shape[1])
    label4, data4 = ut.loadLocalData(sPath=path4, bHeader=None)
    # print(data4.shape[0], data4.shape[1])
    label5, data5 = ut.loadLocalData(sPath=path5, bHeader=None)
    print(data5.shape[0], data5.shape[1])

    for i in range(0, len(label5)):
        print(i, label1[i], label5[i])

    data = np.concatenate((data1, data2), axis=0)
    # print(data.shape[0], data.shape[1])
    data = np.concatenate((data, data3), axis=0)
    # print(data.shape[0], data.shape[1])
    data = np.concatenate((data, data4), axis=0)
    print(data.shape[0], data.shape[1])

    df_data = pd.DataFrame(data=data, columns=label1)
    df_data5 = pd.DataFrame(data=data5, columns=label5)
    df_data = pd.concat([df_data, df_data5], axis=0, ignore_index=True)
    print(df_data)


def main():
    # localPath = 'C:/Users/uic54751/Desktop/Thesis/ML/Data/Innerliner03/InfluxDB_Innerliner03_20112020.csv'
    sLocalPath = '.csv'
    bStartTimer = True
    bLoadLocalData = True
    bGetInfluxData = False
    bDebugMain = True
    bDebugProcess = True
    bScaleDataColumn = True
    bClearLabels = True
    if(bLoadLocalData):
        labels, processed_data = ut.utilsLoop(bStartTimer=bStartTimer,
                                              bLoadLocalData=bLoadLocalData,
                                              sLocalDataPath=sLocalPath,
                                              bGetInfluxData=bGetInfluxData,
                                              bDebugMain=bDebugMain,
                                              bDebugProcess=bDebugProcess,
                                              bScaleDataColumn=bScaleDataColumn,
                                              bClearLabels=bClearLabels)
    elif(bLoadLocalData and bGetInfluxData):
        labels, processed_data, influx_unprocessed_data = ut.utilsLoop(bStartTimer=bStartTimer,
                                              bLoadLocalData=bLoadLocalData,
                                              sLocalDataPath=sLocalPath,
                                              bGetInfluxData=bGetInfluxData,
                                              bDebugMain=bDebugMain,
                                              bDebugProcess=bDebugProcess,
                                              bScaleDataColumn=bScaleDataColumn,
                                              bClearLabels=bClearLabels)

    # Numpy array to Pandas dataframe
    df_processed_data = pd.DataFrame(data= processed_data, columns=labels)
    # drop columns of string type
    df_processed_data = df_processed_data.drop(columns=['time', 'Machine']) # 'string_Recipe','','',''])

    print(df_processed_data)
    # print(df_processed_data.dtypes)
    # print(df_processed_data.astype(float))
    # print(df_processed_data.dtypes)

    df_processed_data = df_processed_data.astype(float)
    # print(df_processed_data.dtypes)
    # df_processed_data.to_csv("data1.csv")


    # Pearson correlation
    overall_pearson_r = df_processed_data.corr(method='pearson')
    # print(overall_pearson_r)

    # plt.figure(figsize=(20, 15), dpi=500)
    # ax = plt.subplot(111)
    # sns.heatmap(overall_pearson_r, annot=False, ax=ax)
    # plt.show()

    # r, p = stats.pearsonr()
    # print(f"Scipy computed Pearson r: {r} and p-value: {p}")

def debug():
    sLocalPath = '.csv'
    bStartTimer = True
    bLoadLocalData = True
    bGetInfluxData = False
    bDebugMain = True
    bDebugProcess = True
    bScaleDataColumn = True
    bClearLabels = True

    # debug load data
    labels, unprocessed_data = ut.loadLocalData(sPath=sLocalPath, bHeader=None)

    labels, processed_data = ut.utilsLoop(bStartTimer=bStartTimer,
                                          bLoadLocalData=bLoadLocalData,
                                          sLocalDataPath=sLocalPath,
                                          bGetInfluxData=bGetInfluxData,
                                          bDebugMain=bDebugMain,
                                          bDebugProcess=bDebugProcess,
                                          bScaleDataColumn=bScaleDataColumn,
                                          bClearLabels=bClearLabels)

    print(ut.bcolors.ENDLINE)

    # print(unprocessed_data[0][0])
    # print(processed_data[0][0])
    # print(unprocessed_data[0][1])
    # print(processed_data[0][1])
    # print(unprocessed_data[0][2])
    # print(processed_data[0][2])

    for i in range(0, unprocessed_data.shape[0]):
        for j in range(0, unprocessed_data.shape[1]):
            print("[" + str(i) + "]" + "[" + str(j) + "] -> " + labels[j] + ": " + str(unprocessed_data[i][j]) + " -- " + str(processed_data[i][j]))
        print("New Line" + ut.bcolors.ENDLINE)


if __name__ == '__main__':
    # main()

    debug()
    # joinLocalData()

    # print(type(tf))
    # ut.print_versions(tf, pd)



