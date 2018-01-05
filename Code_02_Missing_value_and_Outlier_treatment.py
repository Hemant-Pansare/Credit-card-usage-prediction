# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:35:48 2017

@author: hemant.pansare
"""

import pandas as pd

def missing_and_outlier_treatment(combined_data):
    #---------------- Missing value imputation and outlier treatment -------------------------
    
    #get % of null values in each column
    combined_data.isnull().sum()/len(combined_data) * 100
    
    # impute missing values with mode
    impute_missing_with_mode = ['emp_ind', 'cust_country_resid', 'gender', 'first_contract_date', 'ind_relation', 'cust_rel_in_beg', 'residence_ind', 'foreigner_ind', 'spouse_ind', 'channel_used_to_join', 'deceased_ind', 'province_code', 'province_name', 'activity_ind', 'cust_type_in_beg', 'segmentation', 'new_cust_ind']
    
    for var in impute_missing_with_mode:
        combined_data[var].fillna(combined_data[var].mode()[0], inplace = True)
    
    #impute missing value with 0
    combined_data['address_type'].fillna(0, inplace = True)
    combined_data['last_dt_pri_cust'].fillna(0, inplace = True)
    
    # cust_type_in_beg contains numbers as well as strings, so converting them to number first and then selecting first elements as code
    # unique values are ['1.0', '3.0', '2.0', '1', '3', '4.0', 'P', '4', '2']
    combined_data['cust_type_in_beg'] = combined_data['cust_type_in_beg'].astype(str) 
    combined_data['cust_type_in_beg'] = combined_data['cust_type_in_beg'].str[0]
    combined_data['cust_type_in_beg'].replace('P', '5', inplace = True)
    
    #removing NA values from age
    combined_data['age'] = combined_data['age'].astype(str)
    combined_data['age'].replace(' NA', 0, inplace = True)
    combined_data['age'] = combined_data['age'].astype(int)
    mean_age = int(round(combined_data['age'].mean()))
    combined_data['age'].replace(0, mean_age, inplace = True)
    combined_data.boxplot('age')
    combined_data[combined_data['age'] > combined_data['age'].quantile(q = 0.99)]['age'].count()/len(combined_data)*100
    combined_data['age'][combined_data['age'] > combined_data['age'].quantile(q = 0.99)] = combined_data['age'].quantile(q = 0.99)
    
    #removing NA value from cust_seniority
    combined_data['cust_seniority'] = combined_data['cust_seniority'].astype(str)
    combined_data['cust_seniority'].replace('     NA', 0, inplace = True)
    combined_data['cust_seniority'] = combined_data['cust_seniority'].astype(int)
    combined_data['cust_seniority'].replace(-999999, 0, inplace = True)  # check
    mean_cust_seniority = int(round(combined_data['cust_seniority'].mean()))
    combined_data['cust_seniority'].replace(0, mean_cust_seniority, inplace = True)
    combined_data.boxplot('cust_seniority')
    
    #creating code for segmentation based on values in colum
    #selecting only the numeric part of the value
    combined_data['segmentation'] = combined_data['segmentation'].str[:2]
    
    #get zip level average of gross income
    combined_data['province_code'] = combined_data['province_code'].astype(int)
    combined_data['gross_income'] = combined_data['gross_income'].astype(str)
    combined_data['gross_income'].replace('         NA', 'NaN', inplace = True)
    combined_data['gross_income'] = combined_data['gross_income'].astype(float)
    zip_gross_income = combined_data[['province_code', 'gross_income']]
    rolledup_gross_income = zip_gross_income.groupby('province_code').median()['gross_income'].rename('median_gross_income')
    rolledup_gross_income = rolledup_gross_income.reset_index()
    
    #missing value imputation for gross_income
    combined_data_final = pd.merge(combined_data, rolledup_gross_income, how = 'inner', left_on = 'province_code', right_on = 'province_code')
    combined_data_final['gross_income'].fillna(combined_data_final['median_gross_income'], inplace = True)
    combined_data_final['gross_income'] = combined_data_final['gross_income'].astype(float)
    del combined_data_final['median_gross_income']
    
    #checking for outliers
    combined_data_final['imputed_gross_income'] = combined_data_final['gross_income']
    combined_data_final.boxplot('imputed_gross_income')
    combined_data_final[combined_data_final['imputed_gross_income'] > combined_data_final['imputed_gross_income'].quantile(q = 0.99)]['imputed_gross_income'].count()/len(combined_data_final)*100
    combined_data_final['imputed_gross_income'][combined_data_final['imputed_gross_income'] > combined_data_final['imputed_gross_income'].quantile(q=0.99)] = combined_data_final['imputed_gross_income'].quantile(q=0.99)
    
    #creating new variable using first_contract_date and maint_dt
    combined_data_final['temp'] = (pd.to_datetime(combined_data_final['maint_dt'], format = '%Y-%m-%d') - pd.to_datetime(combined_data_final['first_contract_date'], format = '%Y-%m-%d')).astype(str)
    combined_data_final['temp1'], combined_data_final['temp2'], combined_data_final['temp3'] = combined_data_final['temp'].str.split(' ').str
    combined_data_final['diff_maintdt_firstcontractdt'] = ((combined_data_final['temp1'].astype(int))/30).astype(int)
    combined_data_final = combined_data_final.drop(['temp', 'temp1', 'temp2', 'temp3'], axis = 1)
    return combined_data_final
