import numpy as np
from datetime import datetime
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re
import sys
import json


def deal_with_date(df):
    """
    The function conduct feature engineer on the datetime feature;
    
    args:
        df (pandas dataframe) : This is the raw data
    
    return:
        df (pandas dataframe) : This is the data after transformation
    """
    def transform_date(x):
        """
        The function parse the string into datetime type
            
        args:
            x (str) : String in the datetime format (%Y-%m-%d %H:%M:%S)
            
        return:
            x (datetime64) : In datetime type
        """
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

def _load_obj(filepath):
        """
    The function load pickle object;
    
    args:
        filepath (str) : Filepath where object store
    
    return:
        pickle object : The available model 
    """

    with open(filepath, 'rb') as f:
        return pickle.load(f)
def map_algorithm(sort_list, sort_interval):
    """
    this algorithm use two pointer algorithm to match two sorted array by Ip address

    args:
        sort_list (List) : sorted list of ip_address
        sort_interval (pandas dataframe): dataframe which contain lower and upper bound of ip address 
                                          as well as a country columns;
    return:
        res (List) : The one after match
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
    """
    This Function map the IP in the df to the Country in Df_IP

    args:
        df (pandas dataframe) : The training data
        df_IP (pandas dataframe): The data have IP interval with country
    return:
        df (pandas dataframe) : The training data with country column
    """

    IP_col = df['ip_address']
    res = map_algorithm(np.array(IP_col), df_IP)
    IP_Country = pd.DataFrame({'IP': IP_col, 'Country': res})
    df = pd.merge(df, IP_Country, left_index=True, right_index=True).drop(columns='ip_address')
    df['Country'].fillna('Unknow', inplace=True)
    return df

def transform_category(df,SETTINGS):
     """
    This Function load the Label Encoder pickle object and convert
    category data in df to numberic one

    args:
        df (pandas dataframe) : The training data
        SETTINGS (Json): The configuration file
    """
   
    for item in SETTINGS['Transformer']:
        CLF=_load_obj(SETTINGS['Transformer'][item])
        df[item]=CLF.transform(df[item])

def _load_setting(path):
    """
    This Function load the configuration file

    args:
        path (str) : name of configuration file
    """

    fp=open(os.path.join('.',path),'rb')
    SETTINGS=json.load(fp)
    fp.close()
    return SETTINGS
    
def main():
    SETTINGS=_load_setting(SETTING=sys.argv[1])
    geo=pd.read_csv(SETTINGS['IP_country_map_path'])
    geo.sort_values('lower_bound_ip_address', ascending=True, inplace=True)
    df=pd.read_csv(SETTINGS['input_path'])
    #map IP
    df=deal_with_IP(df,geo)
    #deal with feature engineer
    df=deal_with_date(df)
    #load label encoder and convert category
    transform_category(df,SETTINGS)
    df.drop(columns=['user_id', 'signup_time', 'purchase_time','device_id','IP'],inplace=True)
    Model=_load_obj(SETTINGS['model_path'])
    if SETTINGS['prediction_type']=='prediction':
        Y_pre = Model.predict(df)
        Y_pre_ = pd.DataFrame(data=Y_pre, columns=['class'])
    elif SETTINGS['prediction_type']=='probability':
        Y_pre = Model.predict_proba(df)
        Y_pre_ = pd.DataFrame(data=Y_pre[:,1], columns=['class'])
    Y_pre_.to_csv(SETTINGS['output_path'])


if __name__ == '__main__':
    main()