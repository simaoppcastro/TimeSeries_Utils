import sys

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import timeit
import time
import datetime
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import influxdb_client
from influxdb import InfluxDBClient
from sklearn import preprocessing

'''
Simão Castro
Versions History:
v0: 11/2020 first functions: load local data, load data via influxdb query
v1: 11/2020 process data: replace NaN values by zero; scaleColArray
v2: 11/2020 scaleColArray: scale the values of a passed column vector/array
v3: 26/11/2020 scaling changes on process data function
v4: 27/11/2020 verify data type
v5: 30/11/2020 main pass to "main.py"
    process version 2 (v2)
note : verify data after processing (on going) 
'''

'''
messages
colors print
'''
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDLINE = FAIL + '-----------------------------------------------------------------------------------------------------------------------------' + ENDC

'''
verify data type
version: v0 label input -> return data type
'''
def verifyDataType(sLabel):
    type = sLabel.split('_')[0]
    # debug return data type
    # print(type)
    return type

'''
clean label
version: v0 label input -> return label name (clean)
version: v1 remove debug print; try exception new label result
'''
def clearLabel(sLabel):
    try:
        label = sLabel.split('_')[1]
    except:
        label = sLabel

    # debug return value
    # print(label)
    return label

'''
version: v1
print lib. versions
'''
def print_versions():
    print("Tensorflow: " + str(tf.__version__))
    print("Numpy" + str(np.__version__))
    print("Pandas" + str(pd.__version__))
    print("Keras: " + str(keras.__version__))

'''
version: v1
load local .csv data
'''
def loadLocalData(sPath, bHeader):
    try:
        unprocessed_dataFrame = pd.read_csv(sPath, header=bHeader, dtype=None)
        unprocessed_data = unprocessed_dataFrame.values
        labels = unprocessed_data[0]
        unprocessed_data = np.delete(unprocessed_data, 0, 0)
        return labels, unprocessed_data
    except:
        print(bcolors.FAIL + ".read_csv ERROR!" + bcolors.ENDC)

    # debug -> dataframe
    # print(type(unprocessed_dataFrame))
    # shape -> 24300 samples, 96 params
    # print(unprocessed_dataFrame.shape)

    # dataframe to numpy array
    # unprocessed_data = unprocessed_dataFrame.values
    # debug -> numpy array
    # print(type(unprocessed_data))
    # first row -> labels/params
    # print(unprocessed_data[0])
    # labels = unprocessed_data[0]
    # shape -> 24301, 96
    # print(unprocessed_data.shape)
    # print(unprocessed_data[1])

    # unprocessed_data = np.delete(unprocessed_data, 0, 0)
    # return labels, unprocessed_data

'''
version: v1
process data: 
    dtype conversion
    nan replace to zero
'''
def processDataOld(npArrayLabels, npArrayData, bScaleDataColumn, bDebugProcess):
    rows = npArrayData.shape[0]
    cols = npArrayData.shape[1]
    if(bDebugProcess):
        print(rows)
        print(cols)

    for i in range(rows):
        if(bDebugProcess):
            print("Row nº: " + str(i))

        for j in range(cols):
            # try convert numbers to floats
            try:
                npArrayData[i, j] = np.float64(npArrayData[i, j])
            except:
                pass

            # verify the existence of 'nan', if so, replace with 0 (zero)
            try:
                if (np.isnan(npArrayData[i, j])): npArrayData[i, j] = 0
            except:
                pass

            if(bDebugProcess):
                print(str(npArrayLabels[j]) + ': ' + str(npArrayData[i, j]))
                print(str(npArrayLabels[j]) + ' (Type): ' + str(type(npArrayData[i, j])))

    if(bScaleDataColumn):
        # count = 0
        for col in range(cols):
            try:
                # debug for verify data type
                # print(npArrayLabels[col])
                if(verifyDataType(npArrayLabels[col]) != 'string'):
                    y_vec_scaled, ymin, ymax = scaleColArray(npArrayData.T[col])
                    npArrayData.T[col] = y_vec_scaled

                # debug scaleColArray result
                if (bDebugProcess or True):
                # if (bDebugProcess):
                    print(str(npArrayLabels[col]) + ': ' + str(npArrayData.T[col]))
                    try:
                        print("Min: " + str(ymin) + "; Max: " + str(ymax))
                    except:
                        # if dont scale ymin and ymax dont exist -> pass exception
                        pass
                    print('------------------------------------------------------------------------------------------------------------------')

            except:
                pass

            # y_vec_scaled, ymin, ymax = scale_array(col)
            # npArrayData.T[0] = y_vec_scaled
            '''
                y_vec = processed_data[:, 4]
                print(y_vec)
                y_vec_scaled, ymin, ymax = scale_array(y_vec)
                print(y_vec_scaled)
            '''

    return npArrayLabels, npArrayData

'''
version: v2
process data: 
    dtype conversion
    nan replace to zero
    scale column
'''
def processData(npArrayLabels, npArrayData, bScaleDataColumn, bDebugProcess, bClearLabels):
    rows = npArrayData.shape[0]
    cols = npArrayData.shape[1]

    # debug
    print('Rows:', rows, ', Cols: ', cols)


    # clear data
    print(bcolors.OKGREEN +  "Cleaning Data ..." + bcolors.ENDC)
    for j in range(cols):
        # Convert to float and scale column
        # if (bDebugProcess): print('Before:', str(npArrayLabels[j]) + ': ' + str(npArrayData.T[j]) + bcolors.OKCYAN)
        if (verifyDataType(npArrayLabels[j]) != 'string'
                and verifyDataType(npArrayLabels[j]) != 'bool'
                and npArrayLabels[j] != 'Machine'
                and npArrayLabels[j] != 'time'):

            # debug before
            # if(bDebugProcess): print(str(npArrayLabels[j]) + ': ' + str(npArrayData.T[j]))

            if (bDebugProcess): print(bcolors.OKCYAN + 'Before:', str(npArrayLabels[j]) + ': ' + str(npArrayData.T[j]) + bcolors.OKCYAN)

            # convert type number to float
            try:
                y_vec_converted = convertColArrayFloat64(npArrayData.T[j])
                npArrayData.T[j] = y_vec_converted
            except:
                if (bDebugProcess): print('Error: ', 'convertColArrayFloat64')

            # verify the existence of 'nan', if so, replace with 0 (zero)
            try:
                y_vec_nan = removeNanColArray(y_vec=npArrayData.T[j], bMean=True, bDebug=bDebugProcess)
                npArrayData.T[j] = y_vec_nan
            except:
                if(bDebugProcess): print('Error: ', 'removeNanColArray')

            if (bScaleDataColumn):
                # scale 0 - 1
                try:
                    y_vec_scaled, ymin, ymax = scaleColArray(npArrayData.T[j])
                    npArrayData.T[j] = y_vec_scaled
                except:
                    if(bDebugProcess): print('Error: ', 'scaleColArray')

            # debug after processing
            if(bDebugProcess): print(bcolors.OKGREEN + 'After: ', str(npArrayLabels[j]) + ': ' + str(npArrayData.T[j]) + bcolors.ENDC)

        # string -> NO Scale and NO convert to float
        # only remove nan values
        elif (verifyDataType(npArrayLabels[j]) == 'string'):
            # debug before
            if(bDebugProcess): print(bcolors.OKCYAN + 'Before:', str(npArrayLabels[j]) + ': ' + str(npArrayData.T[j]) + bcolors.ENDC)

            # verify the existence of 'nan', if so, replace with 0 (zero)
            y_vec_nan = removeNanColArray(y_vec=npArrayData.T[j], bMean=False, bDebug=bDebugProcess)
            npArrayData.T[j] = y_vec_nan

            # debug after
            if(bDebugProcess): print(bcolors.OKGREEN + 'After: ', str(npArrayLabels[j]) + ': ' + str(npArrayData.T[j]) + bcolors.ENDC)

        # bool -> NO Scale and NO convert to float
        # only remove nan values
        # and convert True to 1 and False to 0
        elif (verifyDataType(npArrayLabels[j]) == 'bool'):
            # debug before
            if (bDebugProcess): print(bcolors.OKCYAN + 'Before:', str(npArrayLabels[j]) + ': ' + str(npArrayData.T[j]) + bcolors.ENDC)

            # verify the existence of 'nan', if so, replace with 0 (zero)
            y_vec_nan = removeNanColArray(y_vec=npArrayData.T[j], bMean=False, bDebug=bDebugProcess)
            npArrayData.T[j] = y_vec_nan

            # process bool array/column
            y_vec_bool = convertBoolArray(y_vec=npArrayData.T[j])
            npArrayData.T[j] = y_vec_bool

            # debug after
            if (bDebugProcess): print(bcolors.OKGREEN + 'After: ', str(npArrayLabels[j]) + ': ' + str(npArrayData.T[j]) + bcolors.ENDC)

    # clear labels
    print(bcolors.OKGREEN +  "Cleaning Data Labels ..." + bcolors.ENDC)
    if (bClearLabels):
        for col in range(0, len(npArrayLabels)):
            if(bDebugProcess): print(bcolors.OKCYAN + "Before: " + str(npArrayLabels[col]) + bcolors.ENDC)
            npArrayLabels[col] = clearLabel(npArrayLabels[col])
            if(bDebugProcess): print(bcolors.OKCYAN + "After: " + str(npArrayLabels[col]) + bcolors.ENDC)

    print(bcolors.OKGREEN +  "Process Data Complete ..." + bcolors.ENDC)
    return npArrayLabels, npArrayData

'''
version: v1
verify the existence of 'nan', if so, replace with 0 (zero)
version: v2
replace with the mean of the column (before this is replaced by zero)
version: v3
replace with mean on columns of numbers/float
and replace with 0 on columns of strings
'''
def removeNanColArray(y_vec, bMean, bDebug = False):
    # debug
    # print('Debug: ', bDebug)
    _y_vec = y_vec
    for k in range(0, len(y_vec)):
        try:
            if (np.isnan(y_vec[k]) and bMean):
                # _y_vec[k] = np.mean(y_vec)
                # https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
                # debug
                # if (bDebug): print('Len: ', len(y_vec), 'Mean: ', np.nanmean(y_vec))
                _y_vec[k] = np.nanmean(y_vec)
                if(bDebug): print("NaN replaced by Mean: " + str(_y_vec[k]))
            elif(np.isnan(y_vec[k]) and not bMean):
                _y_vec[k] = 0
                if(bDebug): print("NaN replaced by Zero ")
        except:
            pass

    return _y_vec

'''
version: v1
scale column of data
'''
def scaleColArray(y_vec):
    # ymax = np.max(y_vec)
    ymax = np.nanmax(y_vec)
    # ymin = np.min(y_vec)
    ymin = np.nanmin(y_vec)


    y_vec_scaled = y_vec

    for k in range(0, len(y_vec)):
        # y_vec_scaled[k][0] = (y_vec[k][0] - ymin) / (ymax - ymin)
        y_vec_scaled[k] = (y_vec[k] - ymin) / (ymax - ymin)

    return y_vec_scaled, ymin, ymax

'''
version: v1
convert number type vars from column (input) to float
convert type number to float
'''
def convertColArrayFloat64(y_vec):
    _y_vec = y_vec
    for k in range(0, len(y_vec)):
        _y_vec[k] = np.float64(y_vec[k])

    return _y_vec

'''
version: v1
process bool array/column
True = 1; False = 0
'''
def convertBoolArray(y_vec):
    _y_vec = y_vec
    for k in range(0, len(y_vec)):
        if(_y_vec[k] == 'True'): _y_vec[k] = 1
        elif(_y_vec[k] == 'True'): _y_vec[k] = 0
        else: _y_vec[k] = 0

    return _y_vec

'''
version: alpha
get informations from data
'''
def testCorrelationsHeatMap(npArrayData, npArrayLabels):
    # nunpy array to pandas dataframe
    dataFrame = pd.DataFrame(data=npArrayData, columns=npArrayLabels)
    # print(dataFrame)
    # corrMatrix = dataFrame.corr()
    # print(corrMatrix)
    # sns.heatmap(corrMatrix, annot=True)
    # sns.heatmap(dataFrame, annot=True)

    # plt.show()
    scaler = preprocessing.MinMaxScaler()

'''
version: v1
influxdb client -> query´s last entry
notes: https://influxdb-python.readthedocs.io/en/latest/examples.html
'''
def getInfluxDbData():
    host = '10.205.34.78'
    port = 8086
    user = 'eng'
    password = 'shopfloor'
    dbname = 'Innerliner03'
    dbuser = ''
    dbuser_password = ''
    query = 'select * from data_measurement WHERE time > now() - 1m;'
    # query = 'select * from data_measurement WHERE time > now() - 5m;'
    # query_where = 'select Int_value from cpu_load_short where host=$host;'

    client = InfluxDBClient(host, port, user, password, dbname)
    # print("Querying data: " + query)
    # result = client.query(query)
    # print("Result: {0}".format(result))

    try:
        result = client.query(query)
        return result
    except:
        print(bcolors.FAIL + "Error InfluxDb Query!")

'''
version: v1
main loop
this function will pass to another program
version v2: main loop -> utilsLoop is called from main.py
'''
def utilsLoop(bStartTimer = False, bLoadLocalData = False, sLocalDataPath = '', bGetInfluxData = False, bDebugMain = True, bDebugProcess = False, bScaleDataColumn = True, bClearLabels = True):
    print(bcolors.HEADER + 'Options: \nTimer = ' + str(bStartTimer)
          + '\nLoadLocalData = ' + str(bLoadLocalData)
          + '\nLocalPath = ' + str(sLocalDataPath)
          + '\nGetInfluxData = ' + str(bGetInfluxData)
          + '\nScaleDataColumn = ' + str(bScaleDataColumn)
          + '\nClearLabels = ' + str(bClearLabels)
          + '\nDebugMain = ' + str(bDebugMain)
          + '\nDebugProcess = ' + str(bDebugProcess)
          + bcolors.ENDC)

    start = timeit.default_timer()
    print(bcolors.OKBLUE + 'Start: ' + str(datetime.datetime.now()) + bcolors.ENDC)

    # load local data
    if (bLoadLocalData):
        try:
            labels, unprocessed_data = loadLocalData(sPath=sLocalDataPath, bHeader=None)
            print(bcolors.OKGREEN + 'Local Data Loaded OK' + bcolors.ENDC)

            try:
                # process local data
                labels, processed_data = processData(npArrayLabels=labels, npArrayData=unprocessed_data,
                                                     bScaleDataColumn=bScaleDataColumn, bDebugProcess=bDebugProcess, bClearLabels=bClearLabels)
                print(bcolors.OKGREEN + 'Local Data Process OK' + bcolors.ENDC)
            except:
                print(bcolors.FAIL + 'Error processData!' + bcolors.ENDC)
        except:
            print(bcolors.FAIL + 'Error loadLocalData!' + bcolors.ENDC)



        # labels, processed_data = processData(npArrayLabels=labels, npArrayData=unprocessed_data,bScaleDataColumn=bScaleDataColumn, bDebugProcess=bDebugProcess, bClearLabels=bClearLabels)

        if (bGetInfluxData):
            try:
                influx_unprocessed_data = getInfluxDbData()
                print(bcolors.OKGREEN + 'Influx Data Loaded' + bcolors.ENDC)
            except:
                print(bcolors.FAIL + 'Error getInfluxDbData!' + bcolors.ENDC)

    timeTaken = (timeit.default_timer()) - start
    print(bcolors.OKBLUE + "Time Spent: " + str(timeTaken) + 's' + bcolors.ENDC)
    print(bcolors.OKBLUE + 'End: ' + str(datetime.datetime.now()) + bcolors.ENDC)

    # return data
    if(bLoadLocalData): return labels, processed_data
    elif(bLoadLocalData and bGetInfluxData): return labels, processed_data, influx_unprocessed_data







