# import pickle
import sys
import numpy as np

def calculate_tmax_RR(Q_residual):
    global maxThreshold1
    maxSum = sys.float_info.max
    max_cadidate = max(Q_residual)
    print(max_cadidate)
    for taoThreshold in np.arange(0.00025, max_cadidate, 0.00025):
        costSum = 0
        for row in Q_residual:
            if (row > 0):
                costSum += abs(taoThreshold - row)
        if (maxSum > costSum):
            maxSum = costSum
            maxThreshold1 = taoThreshold
    return maxThreshold1

def calculate_tmin_RR(Q_residual):
    global minThreshold1
    minSum = sys.float_info.max
    min_candidate = min(Q_residual)
    print(min_candidate)
    for taoThreshold in np.arange(min_candidate, 0, 0.00025):
        costSum = 0
        for row in Q_residual:
            if (row < 0):
                costSum += abs(taoThreshold - row)
        if (minSum > costSum):
            minSum = costSum
            minThreshold1 = taoThreshold
    return minThreshold1

def old_calculate_tmax(Q_residual):
    global maxThreshold1
    maxSum = sys.float_info.max
    max_cadidate = max(Q_residual)
    print(max_cadidate)
    for taoThreshold in np.arange(0.00025, max_cadidate, 0.00025):
        costSum = 0
        pSum = 0
        for row in Q_residual:
            if (row > 0):
                if (row <  taoThreshold):
                    costSum += abs(taoThreshold - row) / 2
                else:
                    pSum += abs(taoThreshold - row) * 2
        taoSumDiff = abs(costSum - pSum)
        if (maxSum > taoSumDiff):
            maxSum = taoSumDiff
            maxThreshold1 = taoThreshold
    return maxThreshold1

def old_calculate_tmin(Q_residual):
    global minThreshold1
    minSum = sys.float_info.max
    min_candidate = min(Q_residual)
    print(min_candidate)
    for taoThreshold in np.arange(min_candidate, 0, 0.00025):
        costSum = 0
        pSum = 0
        for row in Q_residual:
            if (row < 0):
                if (row > taoThreshold):
                    costSum += abs(taoThreshold - row) / 2
                else:
                    pSum += abs(taoThreshold - row) * 2
        taoSumDiff = abs(costSum - pSum)
        if (minSum > taoSumDiff):
            minSum = taoSumDiff
            minThreshold1 = taoThreshold
    return minThreshold1

def calculate_tmax(Q_residual):
    max_candidate = max(Q_residual)
    thresholds = list(np.arange(0.00025, max_candidate, 0.00025))
    if not thresholds:
        return None
    
    thresholds_mat = np.repeat([np.array(thresholds)], len(Q_residual), axis=0)
    residuals_mat = np.repeat([Q_residual.to_numpy()], len(thresholds), axis=0).transpose()
    
    costSum = np.where(((residuals_mat > 0) & (residuals_mat < thresholds_mat)), 
                         abs(thresholds_mat - residuals_mat) / 2, 
                         0)
    
    pSum    = np.where(((residuals_mat > 0) & ~(residuals_mat < thresholds_mat)), 
                         abs(thresholds_mat - residuals_mat) * 2, 
                         0)
    
    taoSumDiff_arr = abs(costSum.sum(axis=0) - pSum.sum(axis=0))
    try:
        min_idx = np.where(taoSumDiff_arr == taoSumDiff_arr.min())[0][0]
        return thresholds[min_idx]
    except ValueError:
        print(thresholds_mat.shape)
        print(residuals_mat.shape)
        print(costSum.shape)
        print(pSum.shape)
        print(taoSumDiff_arr.shape)
        return None
    
def calculate_tmin(Q_residual):
    min_candidate = min(Q_residual)
    thresholds = list(np.arange(min_candidate, 0, 0.00025))
    if not thresholds:
        return None
    
    thresholds_mat = np.repeat([np.array(thresholds)], len(Q_residual), axis=0)
    residuals_mat = np.repeat([Q_residual.to_numpy()], len(thresholds), axis=0).transpose()
    
    costSum = np.where(((residuals_mat < 0) & (residuals_mat > thresholds_mat)), 
                         abs(thresholds_mat - residuals_mat) / 2, 
                         0)
    
    pSum    = np.where(((residuals_mat < 0) & ~(residuals_mat > thresholds_mat)), 
                         abs(thresholds_mat - residuals_mat) * 2, 
                         0)
    
    taoSumDiff_arr = abs(costSum.sum(axis=0) - pSum.sum(axis=0))
    try:
        min_idx = np.where(taoSumDiff_arr == taoSumDiff_arr.min())[0][0]
        return thresholds[min_idx]
    except ValueError:
        print(thresholds_mat.shape)
        print(residuals_mat.shape)
        print(costSum.shape)
        print(pSum.shape)
        print(taoSumDiff_arr.shape)
        return None
        
def calculate_tmax_huber(Q_residual,beta,w1,w2):
    max_candidate = max(Q_residual)
    thresholds = list(np.arange(0.00025, max_candidate, 0.00025))
    if not thresholds:
        return None
    thresholds_mat = np.repeat([np.array(thresholds)], len(Q_residual), axis=0)
    residuals_mat = np.repeat([Q_residual.to_numpy()], len(thresholds), axis=0).transpose()

    costSum = np.where(((residuals_mat > 0) & ((thresholds_mat - residuals_mat) > 0)),
                       pow((thresholds_mat - residuals_mat)*w2,2)*(0.5),beta*abs((thresholds_mat - residuals_mat))*w2 - (0.5)*pow(beta,2))

    pSum = np.where(((residuals_mat > 0) & ((thresholds_mat - residuals_mat) < 0)),
                    pow((thresholds_mat - residuals_mat) * w1, 2) * (0.5),beta * abs((thresholds_mat - residuals_mat)) * w1 - (0.5) * pow(beta, 2))

    taoSumDiff_arr = abs(costSum.sum(axis=0) + pSum.sum(axis=0))
    try:
        min_idx = np.where(taoSumDiff_arr == taoSumDiff_arr.min())[0][0]
        return thresholds[min_idx]
    except ValueError:
        print(thresholds_mat.shape)
        print(residuals_mat.shape)
        print(costSum.shape)
        print(pSum.shape)
        print(taoSumDiff_arr.shape)
        return None

def calculate_tmin_huber(Q_residual,beta,w1,w2):
    max_candidate = max(Q_residual)
    thresholds = list(np.arange(0.00025, max_candidate, 0.00025))
    if not thresholds:
        return None
    thresholds_mat = np.repeat([np.array(thresholds)], len(Q_residual), axis=0)
    residuals_mat = np.repeat([Q_residual.to_numpy()], len(thresholds), axis=0).transpose()
    
    costSum = np.where(((residuals_mat < 0) & ((thresholds_mat - residuals_mat) > 0)),
                       pow((thresholds_mat - residuals_mat) * w2, 2) * (0.5),
                       beta * abs((thresholds_mat - residuals_mat)) * w2 - (0.5) * pow(beta, 2))

    pSum = np.where(((residuals_mat < 0) & ((thresholds_mat - residuals_mat) < 0)),
                    pow((thresholds_mat - residuals_mat) * w1, 2) * (0.5),
                    beta * abs((thresholds_mat - residuals_mat)) * w1 - (0.5) * pow(beta, 2))

    taoSumDiff_arr = abs(costSum.sum(axis=0) + pSum.sum(axis=0))
    try:
        min_idx = np.where(taoSumDiff_arr == taoSumDiff_arr.min())[0][0]
        return thresholds[min_idx]
    except ValueError:
        print(thresholds_mat.shape)
        print(residuals_mat.shape)
        print(costSum.shape)
        print(pSum.shape)
        print(taoSumDiff_arr.shape)
        return None

#endregion
def calculate_nabla(grouped_frame_of_focus,safe_margin):
    nabla_time_indexed = {}
    for key, group in grouped_frame_of_focus:
        for index,item in group.iteritems():
            if(item > safe_margin['upper'][key]):
                nabla_time_indexed[index] = (item -safe_margin['upper'][key])
            elif(item < safe_margin['lower'][key]):
                nabla_time_indexed[index] = (item -safe_margin['lower'][key])
            else:
                nabla_time_indexed[index] = 0
    return nabla_time_indexed

def faster_calculate_residual(nabla, SF):
    RUC = {}
    grouped = nabla.groupby(nabla.index.floor('D'))
    for k, group in grouped:
        df = group.rolling(SF, min_periods=SF).sum()
        df[0:SF] = group[0:SF]
        _RUC = df.to_dict()['nabla']
        RUC.update(_RUC)
    return RUC

def calculate_residual(nabla,SF):
    RUC = {}
    # print("this is executing")
    nabla_frame_grouped_by_year = nabla.groupby([nabla.index.year])
    for key_Y,group_Y in nabla_frame_grouped_by_year:
        # print(group_Y)
        grouped_by_month = group_Y.groupby([group_Y.index.month])
        for key_M, group_M in grouped_by_month:
            # print(group_M)
            grouped_by_day= group_M.groupby([group_M.index.day])
            for key_D,group_D  in grouped_by_day:
                frame_iterator = 0
                # print(group_D)
                for index,item in group_D.iterrows():
                    if(frame_iterator >= SF):
                        temp_df = group_D[group_D.index<=index].tail(SF)
                        # print(frame_iterator)
                        res_sum_0 = 0
                        # print("New Cycle")
                        for index1,item1 in temp_df.iterrows():
                            # print('Before: ',res_sum_0)
                            res_sum_0 = res_sum_0 +  temp_df.loc[index1,'nabla']
                            # print('After adding: ',temp_df.loc[index1,'nabla'],' Sum: ',res_sum_0)
                        # print(res_sum_0)
                        RUC[index] = res_sum_0
                    else:
                        RUC[index] = group_D.loc[index,'nabla']
                    frame_iterator = frame_iterator + 1
    return RUC