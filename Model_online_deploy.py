def deal_with_date(df):
    def transform_date(x):
        if not pd.isnull(x):
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        else:
            return x

    df['signup_time'] = df['signup_time'].apply(lambda x: transform_date(x))
    df['purchase_time'] = df['purchase_time'].apply(lambda x: transform_date(x))
    df['sinup_purchase_diff_hour'] = (df['purchase_time'] - df['signup_time']).apply(
        lambda x: x.total_seconds() / 3600)
    # add weekend
    # 5 6 are weekend
    df['signup_time_weekday'] = df['signup_time'].apply(lambda x: x.weekday())
    df['signup_time_is_weekday'] = df['signup_time_weekday'].apply(lambda x: 1 if x == 5 or x == 6 else 0)
    # add weekend
    # 5 6 are weekend
    df['purchase_time_weekday'] = df['purchase_time'].apply(lambda x: x.weekday())
    df['purchase_time_is_weekday'] = df['purchase_time_weekday'].apply(lambda x: 1 if x == 5 or x == 6 else 0)
    df['signup_time_hour'] = df['signup_time'].apply(lambda x: x.hour)
    df['purchase_time_hour'] = df['purchase_time'].apply(lambda x: x.hour)
    # df.drop(columns=[])
    return df

def load_obj(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
def map_algorithm(sort_list, sort_interval):
    """
    this algorithm use the concept of dynamic programming,
    it will iterate over the sort_list one time, for each item, it will iterate over the sort_interval.
    But the tricky part is that this algorithm will only iterate over sort_interval for only one time. once the
    item in sort_list is larger than the upper bound of current interval,
    the algorithm will discard this interval and process to the next one.

    ###############################
    parameter

    sort_list:
    sorted list of ip_address

    sort_interval:
    dataframe which contain lower and upper bound of ip address as well as a country columns
    make sure to sort dataframe by lower bound ip address first
    """
    j = 0
    res = []

    for i in range(len(sort_list)):
        cur_interval = [sort_interval['lower_bound_ip_address'][j], sort_interval['upper_bound_ip_address'][j]]
        cur_country = sort_interval['country'][j]
        #         print(sort_list[i],cur_interval)

        while j < len(sort_interval):
            cur_interval = [sort_interval['lower_bound_ip_address'][j], sort_interval['upper_bound_ip_address'][j]]
            cur_country = sort_interval['country'][j]
            if sort_list[i] < cur_interval[0]:
                res.append(np.nan)
                break
            elif sort_list[i] >= cur_interval[0] and sort_list[i] <= cur_interval[1]:
                res.append(cur_country)
                break
            elif sort_list[i] > cur_interval[1]:
                j += 1
        if j >= len(
                sort_interval):  # this is the early stop method, once the algorithm reach the last interval, then the
            # following data in sort_list will be nan
            break
    res += [np.nan] * (len(sort_list) - len(res))
    return res
def deal_with_IP(df,df_IP):
    IP_col = df['ip_address']
    res = map_algorithm(np.array(IP_col), df_IP)
    IP_Country = pd.DataFrame({'IP': IP_col, 'Country': res})
    df = pd.merge(df, IP_Country, left_index=True, right_index=True).drop(columns='ip_address')
    df['Country'].fillna('Unknow', inplace=True)
    return df

# def deal_with_device(df,dict_):
#     # def convert()
#     df['number_fraude']=df['device_id'].apply(lambda x:dict_[x] if x in dict_ else 0)#for the device not in data, we treat them as naive
#     return
def transform_category(df,args):
    files=os.listdir(args.Working_direction)
    pattern = re.compile('LB')
    LBs=[]
    for line in files:
        if pattern.findall(line):
            LBs.append(line)
    for item in LBs:
        col=item.split(' ')[0]
        CLF=load_obj(item)
        df[col]=CLF.transform(df[col])




if __name__ == '__main__':
    import numpy as np
    from datetime import datetime
    import argparse
    import pickle
    import os
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    import re
    # import xgboost
    parser=argparse.ArgumentParser(description='Slalom online prediction')
    parser.add_argument('--Working_direction', type=str, default='C:/users/lzc/desktop/slalom', help='where all the files store')
    parser.add_argument('--DATA_NAME',type=str,default='test.csv',help='data used to predict')
    parser.add_argument('--MODEL_FILENAME',type=str,default='model.pkl',help='Model used to predict')
    parser.add_argument('--PREDICTION', type=str, default='prediction', help='What type of output you want. Choose probability or prediction. If probability, it will give probability for the class 1')
    parser.add_argument('--OUTPUT_NAME',type=str,default='output.csv',help='file used to store the output')
    parser.add_argument('--IP_Country_NAME', type=str, default='IpAddress_to_Country.csv', help='dictinary have the device-info data')
    args=parser.parse_args()
    os.chdir(args.Working_direction)
    geo=pd.read_csv(args.IP_Country_NAME)
    geo.sort_values('lower_bound_ip_address', ascending=True, inplace=True)
    df=pd.read_csv(args.DATA_NAME)
    #map IP
    df=deal_with_IP(df,geo)
    #datetime
    df=deal_with_date(df)
    #convert category
    transform_category(df,args)
    #model prediction
    df.drop(columns=['user_id', 'signup_time', 'purchase_time','device_id','IP'],inplace=True)
    Model=load_obj(args.MODEL_FILENAME)
    # print(df.columns)
    if args.PREDICTION=='prediction':
        Y_pre = Model.predict(df)
        Y_pre_ = pd.DataFrame(data=Y_pre, columns=['class'])
        Y_pre_.to_csv(args.OUTPUT_NAME)
    elif args.PREDICTION=='probability':
        Y_pre = Model.predict_proba(df)
        Y_pre_ = pd.DataFrame(data=Y_pre[:,1], columns=['class'])
        Y_pre_.to_csv(args.OUTPUT_NAME)



