# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 16:14:56 2017

@author: hemant.pansare
"""

#import required packages
import numpy as np
import pandas as pd
#import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
#from sklearn import cross_validation
from sklearn import metrics
from sklearn.feature_selection import chi2, f_classif, RFE
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import product
import Code_02_Missing_value_and_Outlier_treatment as code02
import Code_03_Visualization as code03
import datetime
import sklearn.grid_search

# dependent variables ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
dependent_variable = ['credit_card', 'short_term_deposits']


#import train and test data
train = pd.read_csv("train_ver2.csv")


#store old column names for future use
train_old_col_names = train.columns


#rename columns
train.rename(columns = {'fecha_dato' : 'maint_dt', 'ncodpers' : 'cust_id', 'ind_empleado' : 'emp_ind', \
                        'pais_residencia' : 'cust_country_resid', 'sexo' : 'gender', 'age' : 'age', \
                        'fecha_alta' : 'first_contract_date', 'ind_nuevo' : 'new_cust_ind', \
                        'antiguedad' : 'cust_seniority', 'indrel' : 'ind_relation', \
                        'ult_fec_cli_1t' : 'last_dt_pri_cust', 'indrel_1mes' : 'cust_type_in_beg', \
                        'tiprel_1mes' : 'cust_rel_in_beg', 'indresi' : 'residence_ind', \
                        'indext' : 'foreigner_ind', 'conyuemp' : 'spouse_ind', \
                        'canal_entrada' : 'channel_used_to_join' , 'indfall' : 'deceased_ind' , \
                        'tipodom' : 'address_type', 'cod_prov' : 'province_code' ,'nomprov' : 'province_name' , \
                        'ind_actividad_cliente' : 'activity_ind', 'renta' : 'gross_income', \
                        'segmento' : 'segmentation', 'ind_tjcr_fin_ult1' : 'credit_card',
                        'ind_deco_fin_ult1' : 'short_term_deposits'},  inplace = True)

# dropping unwanted products from the table
train = train.drop(['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', \
           'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', \
           'ind_ctpp_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', \
           'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', \
           'ind_reca_fin_ult1','ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 
           'ind_nom_pens_ult1', 'ind_recibo_ult1'], axis = 1)


# checking maint_dt from which customers data appears in table
temp = train.copy()
temp1 = temp[['cust_id', 'maint_dt']]
temp1 = temp1.drop_duplicates('cust_id', 'first')
temp2 = pd.DataFrame(temp1.groupby('maint_dt')['cust_id'].count()).reset_index()
print(temp2)
del [temp, temp1, temp2]






# ----------- Run missing value imputation and outlier treatment script ---------------------
combined_data = train.copy()
combined_data_final = code02.missing_and_outlier_treatment(combined_data)



#------------------ creating lag variables for credit card and short term deposits -----------------

temp = combined_data_final.copy()
temp1 = temp[['cust_id', 'maint_dt', 'credit_card', 'short_term_deposits']]
temp2 = temp1.sort(['cust_id', 'maint_dt'])

for lag in range(6):
    temp2['lag_cc_' + str(lag+1)] = temp2['credit_card'].shift(lag+1).fillna(0.0).astype(int)
    
for lag in range(6):
    temp2['lag_std_' + str(lag+1)] = temp2['short_term_deposits'].shift(lag+1).fillna(0.0).astype(int)

df_lag_variables = temp2.copy()
del [temp, temp1, temp2]

temp  = pd.merge(combined_data_final, df_lag_variables, how = 'inner', left_on = ['cust_id', 'maint_dt'], right_on = ['cust_id', 'maint_dt']).drop(['credit_card_y', 'short_term_deposits_y'], axis = 1).rename(columns ={'credit_card_x' : 'credit_card', 'short_term_deposits_x' : 'short_term_deposits'})
combined_data_final = temp.copy()
del [temp, df_lag_variables]

# separate data into training and validation set for credit card data
train_temp1, val_temp1 = combined_data_final[combined_data_final['maint_dt'] < '2016-05-28'], combined_data_final[combined_data_final['maint_dt'] == '2016-05-28']

# ------------ Deduplication of data for dependent variables ---------------
# deduplication of data at cust_id level for dependent variable 'credit_card'
temp1, temp2 = train_temp1[train_temp1['credit_card'] == 1], train_temp1[train_temp1['credit_card'] == 0]
temp1_new = temp1.drop_duplicates('cust_id', 'last')
temp2_new = temp2.drop_duplicates('cust_id', 'last')

temp3 = temp1_new.append(temp2_new)
cc_train_data = temp3.drop_duplicates('cust_id', 'first')
cc_val_data = val_temp1.copy()
del cc_val_data['short_term_deposits']
del [temp1, temp2, temp1_new, temp2_new, temp3]


# deduplication of data at cust_id level for dependent variable 'short_term_deposits'
temp1, temp2 = train_temp1[train_temp1['short_term_deposits'] == 1], train_temp1[train_temp1['short_term_deposits'] == 0]
temp1_new = temp1.drop_duplicates('cust_id', 'last')
temp2_new = temp2.drop_duplicates('cust_id', 'last')

temp3 = temp1_new.append(temp2_new)
std_train_data = temp3.drop_duplicates('cust_id', 'first')
std_val_data = val_temp1.copy()
del std_val_data['credit_card']
del [temp1, temp2, temp1_new, temp2_new, temp3]


# -----  creating visualization for credit card as dependent variable ---------------------------
temp = cc_train_data.copy()
dep_var = dependent_variable[0]
code03.visualization(temp, dep_var)
print(cc_train_data['credit_card'].mean()*100)
# 6.10%

# -----  creating visualization for short term deposits as dependent variables ------------------
temp = std_train_data.copy()
dep_var = dependent_variable[1]
code03.visualization(temp, dep_var)
print(std_train_data['short_term_deposits'].mean()*100)
# 0.70%



















