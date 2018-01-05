# -*- coding: utf-8 -*-
"""
This script creates visualization for independent variables with respect to dependent variables
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def visualization(temp, dep_var):
    # segregating all variables into different lists
    categorical_variables = ['emp_ind', 'cust_country_resid', 'gender', 'new_cust_ind', 'ind_relation', \
                             'cust_type_in_beg', 'cust_rel_in_beg', 'residence_ind', 'foreigner_ind', \
                             'spouse_ind', 'channel_used_to_join', 'deceased_ind', 'address_type',\
                             'province_code', 'province_name', 'activity_ind', 'segmentation']
    
    continuous_variables = ['age', 'cust_seniority', 'imputed_gross_income', 'diff_maintdt_firstcontractdt']
    other_variables = ['first_contract_date', 'last_dt_pri_cust']
    
    
    # creating csv files for categorical variables for visualization.
    for var in categorical_variables:
        print(var)
        temp1 = temp[[var, dep_var]]
        temp2 = pd.DataFrame(temp1.groupby(var).count()).reset_index()
        temp2.columns = [var, 'Frequency']
        temp3 = pd.DataFrame(temp1.groupby(var)[dep_var].sum()).reset_index()
        temp3.columns = [var, 'Response']
        temp4 = pd.merge(temp2, temp3, how = 'inner', left_on = var, right_on = var)
        temp4['Response_rate'] = temp4['Response']/temp4['Frequency']
        del temp4['Response']
        temp4.to_csv('F:\F Drive\Hemant Pansare\Python\Kaggle\Visualization\Export\\' + dep_var + '_' + var + '.csv')
        del [temp1, temp2, temp3, temp4]
    
    
    # creating csv files for continuous variables for visualization
    for var in continuous_variables:
        print(var)
        temp1 = temp[[var, dep_var]]
        temp2 = pd.DataFrame(temp1.groupby(pd.qcut(temp1[var], 10))[dep_var].count()).reset_index()
        temp2.columns = [var, 'Frequency']
        temp3 = pd.DataFrame(temp1.groupby(pd.qcut(temp1[var], 10))[dep_var].sum()).reset_index()
        temp3.columns = [var, 'Response']
        temp4 = pd.merge(temp2, temp3, how = 'inner', left_on = var, right_on = var)
        temp4['Response_rate'] = temp4['Response']/temp4['Frequency']
        del temp4['Response']
        temp4.to_csv('F:\F Drive\Hemant Pansare\Python\Kaggle\Visualization\Export\\' + dep_var + '_' + var + '.csv')
        del [temp1, temp2, temp3, temp4]
    
    
    
