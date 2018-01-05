# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:58:28 2017

@author: hemant.pansare
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_selection import chi2, f_classif, RFE
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import product
import datetime
import sklearn.grid_search


# -------------------------- Building model for credit card usage prediction -----------------------
# ---- creating derived variables for credit card ----------------------
cc_train_data['train_or_val'] = 'train'
cc_val_data['train_or_val'] = 'val'
cc_data = cc_train_data.append(cc_val_data)


# ---- Derived variable creation --------------
cc_data['age_le_30'] = np.where(cc_data['age'] < 30, 1 ,0)
cc_data['age_gt_40_lt_65'] = np.where((cc_data['age'] > 40) & (cc_data['age'] < 65), 1 ,0)
cc_data['age_gt_30_lt_65'] = np.where((cc_data['age'] > 30) & (cc_data['age'] < 65), 1 ,0)

cc_data['channel_used_to_join_grp1'] = np.where((cc_data['channel_used_to_join'].str[:2].isin(['KA'])) | \
                                                (cc_data['channel_used_to_join'].str[2].isin(['4', '7'])), 1, 0)
cc_data['channel_used_to_join_grp2'] = np.where((cc_data['channel_used_to_join'].str[:2].isin(['KA', 'KB', 'KC', 'KF', 'KG', 'RE'])) | \
                                                (cc_data['channel_used_to_join'].str[:].isin(['004', '007', '013'])), 1, 0)

cc_data['cust_rel_in_beg_A'] = np.where(cc_data['cust_rel_in_beg'] == 'A', 1 ,0)

cc_data['cust_seniority_le_60'] = np.where(cc_data['cust_seniority'] <= 60, 1 ,0)
cc_data['cust_seniority_ge_120'] = np.where(cc_data['cust_seniority'] >= 120, 1 ,0)
cc_data['cust_seniority_ge_180'] = np.where(cc_data['cust_seniority'] >= 180, 1 ,0)

cc_data['gender_V'] = np.where(cc_data['gender'] == 'V', 1 ,0)

cc_data['income_le_100k'] = np.where(cc_data['imputed_gross_income'] <= 100000, 1 ,0)
cc_data['income_ge_150k'] = np.where(cc_data['imputed_gross_income'] >= 150000, 1 ,0)

cc_data['province_code_grp1'] = np.where(cc_data['province_code'].isin([2, 3, 4, 5, 6, 9, 10, 	\
                                                                                11, 12, 13, 14, 15, 16, \
                                                                                17, 18, 21, 22, 23, 24, \
                                                                                25, 26, 27, 29, 30, 31, \
                                                                                32, 33, 34, 36, 37, 42, \
                                                                                43, 44, 45, 46, 47, 49, 50]), 1, 0) 
cc_data['province_code_grp2'] = np.where(cc_data['province_code'].isin([2, 5, 6, 10, 15, 16, 21, \
                                                                                25, 27, 30, 32, 34, 36, 37, \
                                                                                44, 49]), 1, 0)
cc_data['province_code_grp3'] = np.where(cc_data['province_code'].isin([28, 38, 52]), 1, 0)

cc_data['segmentation_eq_3'] = np.where(cc_data['segmentation'] == '03', 1, 0)

cc_data['diff_le60'] = np.where(cc_data['diff_maintdt_firstcontractdt'] <= 60, 1 ,0)
cc_data['diff_ge120'] = np.where(cc_data['diff_maintdt_firstcontractdt'] >= 120, 1 ,0)
cc_data['diff_ge180'] = np.where(cc_data['diff_maintdt_firstcontractdt'] >= 180, 1 ,0)

#  ----- separating training and validation data for credit card ---------------
cc_train_data, cc_val_data = cc_data[cc_data['train_or_val'] == 'train'], cc_data[cc_data['train_or_val'] == 'val']





# --------------- Running Iterations for credit card ---------------------------------------------
#cc_shortlisted_variables = ['activity_index', 'age_le_30', 'age_gt_40_lt_65', 'age_gt_30_lt_65', \
#                            'channel_used_to_join_grp1', 'channel_used_to_join_grp2', 'cust_rel_in_beg_A', \
#                            'cust_seniority', 'cust_seniority_le_60', 'cust_seniority_ge_120', \
#                            'cust_seniority_ge_180', 'gender_V', 'imputed_gross_income', \
#                            'income_le_100k', 'income_ge_150k', 'province_code_grp1', 'province_code_grp2', \
#                            'province_code_grp3', 'segmentation_eq_3']

cc_shortlisted_variables = [  'age_gt_40_lt_65','cust_seniority_ge_120',  \
                            'channel_used_to_join_grp1',  'cust_rel_in_beg_A', \
                             'income_le_100k',  \
                             'gender_V',  \
                             'income_ge_150k',  'province_code_grp2', \
                            'province_code_grp3', 'segmentation_eq_3', \
                              'diff_ge180', \
                            'lag_cc_1',  'lag_cc_6']                       
#variables removed 
# ['imputed_gross_income', 'cust_seniority', 'age_le_30', 'activity_index', 'channel_used_to_join_grp2',\
# 'cust_seniority_le_60', 'age_gt_30_lt_65', 'province_code_grp1', 'cust_seniority_ge_180', \
# 'diff_ge120', 'lag_cc_4', 'lag_cc_2', 'lag_cc_5', 'diff_le60', ]                            
#  creating empty dataframe
#summary = pd.DataFrame(np.nan, index = list(range(len(cc_shortlisted_variables))), columns = ['variable', 'training coeff', 'validation coeff', 'Beta stability', 'VIF', 'train p values', 'val p values', 'chi-square'])                            

l = []

cc_x_train = cc_train_data[cc_shortlisted_variables]
cc_y_train = cc_train_data['credit_card']
cc_x_val = cc_val_data[cc_shortlisted_variables]
cc_y_val = cc_val_data['credit_card']

training_model = linear_model.LogisticRegression(class_weight = 'balanced')
training_model.fit(cc_x_train, cc_y_train)

validation_model = linear_model.LogisticRegression(class_weight = 'balanced')
validation_model.fit(cc_x_val, cc_y_val)

train_scores, train_pvalues = chi2(cc_x_train, cc_y_train)
val_scores, val_pvalues = chi2(cc_x_val, cc_y_val)

for i in range(len(cc_shortlisted_variables)):
    l.append([cc_shortlisted_variables[i], training_model.coef_[0][i], \
          validation_model.coef_[0][i], validation_model.coef_[0][i]/training_model.coef_[0][i], \
          variance_inflation_factor(cc_x_train.values, cc_x_train.columns.get_loc(cc_shortlisted_variables[i])), \
          train_pvalues[i], val_pvalues[i], train_scores[i]])   

summary = pd.DataFrame(l, columns = ['variable', 'training coeff', 'validation coeff', 'Beta stability', 'VIF', 'train p values', 'val p values', 'chi-square'])
summary['variable importance'] = summary['chi-square']/np.sum(summary['chi-square'])*100
summary = summary.sort('variable importance', ascending = False).reset_index().drop('index', axis = 1)
summary.to_csv('F:\F Drive\Hemant Pansare\Python\Kaggle\cc_iterations\summary_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.csv')



#---- running grid search -----
logistic_first_model = linear_model.LogisticRegression(random_state = 1)
logistic_params = [{
              'C' : [0.2, 0.4, 0.5, 0.8, 1],
              'class_weight' : ['balanced', {0 : 0.1, 1 : 0.9}, {0 : 0.2, 1 : 0.8}, {0 : 0.25, 1 : 0.75}],
              'max_iter' :  [100, 150, 200, 250, 300]  
              }]
             
best_logistic_model = sklearn.grid_search.GridSearchCV(logistic_first_model, logistic_params, scoring = 'accuracy', cv= 5, refit = True, verbose = 2, n_jobs = 4)
best_logistic_model.fit(cc_x_train, cc_y_train) 

logit_predictions = best_logistic_model.predict(cc_x_val)


#----- running finalized model and predicting values -------------------------
logistic_model = linear_model.LogisticRegression(class_weight = 'balanced')
logistic_model.fit(cc_x_train, cc_y_train)

logistics_predictions = logistic_model.predict(cc_x_val)

np.unique(logistics_predictions, return_counts = True)
fprlogit, tprlogit, thresholds_logit = metrics.roc_curve(cc_y_val, logistics_predictions)
roc_auc_logit = metrics.auc(fprlogit, tprlogit)
print(roc_auc_logit)
#0.829456726787
#0.965553423696 -- with lag variables
#0.952109103768 -- without lag_cc_3 variable

#ploting roc curve
plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.plot(fprlogit, tprlogit, 'b', label = 'AUC = %0.2f' %roc_auc_logit)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



print(2*roc_auc_logit - 1)
#0.658913453575
#0.931106847392  --with lag variables
# 0.904218207537 -- without lag_cc_3 variables

class_combination_logit = pd.DataFrame(cc_y_val.values, logistics_predictions).reset_index().rename(columns = {'index' : 'logistics_predictions', 0 : 'y_validate'}).groupby(['logistics_predictions', 'y_validate']).size().reset_index().rename(columns = {0 : 'Frequency'})
print(class_combination_logit)
pd.pivot_table(class_combination_logit, values = 'Frequency', index = 'logistics_predictions', columns = 'y_validate')
#y_validate                  0      1
#logistics_predictions               
#0                      640781   1941
#1                      255850  32881
#TPR : 94.42%
#FPR : 28.53%

# --with lag variables
#y_validate                  0      1
#logistics_predictions               
#0                      879405   1730
#1                       17226  33092
# TPR : 95.03%
# FPR : 1.92%


#y_validate                  0      1
#logistics_predictions               
#0                      863175   2036
#1                       33456  32786
#
#TPR : 94.15%
#FPR : 3.73%


# ---- ks statistics -------------
logit_prob_predictions = logistic_model.predict_proba(cc_x_val)
logit_actual_and_probabilities = pd.DataFrame(logit_prob_predictions, cc_y_val.values).reset_index().rename(columns = {'index' : 'y_validate', 0 : 'zero_probability', 1 : 'one_probability'})
logit_ks_stat_data = logit_actual_and_probabilities[['y_validate', 'one_probability']]
logit_ks_stat_data = logit_ks_stat_data.sort('one_probability', ascending = False)
logit_ks_stat_data ['deciles'] = pd.qcut(logit_ks_stat_data['one_probability'].rank(method = 'first'), 10, labels = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
number_of_responders = logit_ks_stat_data.groupby('deciles')['y_validate'].sum()
total_pop = logit_ks_stat_data.groupby('deciles')['y_validate'].count()
avg_prob_per_decile = logit_ks_stat_data.groupby('deciles')['one_probability'].mean()


rf_prob_predictions = rf.predict_proba(x_rf_train)
rf_actual_and_probabilities = pd.DataFrame(rf_prob_predictions, y_rf_train.values).reset_index().rename(columns = {'index' : 'y_validate', 0 : 'zero_probability', 1 : 'one_probability'})
rf_ks_stat_data = rf_actual_and_probabilities[['y_validate', 'one_probability']]
rf_ks_stat_data = rf_ks_stat_data.sort('one_probability', ascending = False)
rf_ks_stat_data ['deciles'] = pd.qcut(rf_ks_stat_data['one_probability'].rank(method = 'first'), 10, labels = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
number_of_responders = rf_ks_stat_data.groupby('deciles')['y_validate'].sum()
total_pop = rf_ks_stat_data.groupby('deciles')['y_validate'].count()












#--------------------- finding concordance and discordance ----------
#logistic_model_proba = linear_model.LogisticRegression(class_weight = 'balanced')
#logistic_model_proba.fit(cc_x_train, cc_y_train)
#
#logistics_prob_predictions = logistic_model_proba.predict_proba(cc_x_val)
#
#actual_and_probabilities = pd.DataFrame(logistics_prob_predictions, cc_y_val.values).reset_index().rename(columns = {'index' : 'y_validate', 0 : 'zero_probability', 1 : 'one_probability'})
#zeroes, ones = list(actual_and_probabilities[actual_and_probabilities['y_validate'] == 0]['one_probability']),list(actual_and_probabilities[actual_and_probabilities['y_validate'] == 1]['one_probability'])
#
#
#def chunks(l, n):
#    for i in range(0, len(l), n):
#        yield l[i : i + n]
#
#zeroes_lists = list(chunks(zeroes, 100))
#
#tied = 0
#concordant = 0
#discordant = 0
#
#for i in range(len(zeroes_lists)): #, zeroes_2, zeroes_3, zeroes_4, zeroes_5, zeroes_6, zeroes_7, zeroes_8, zeroes_9]:
#    all_combinations = pd.DataFrame(list(product(zeroes_lists[i] , ones)), columns = ['zeroes', 'ones'])
#
#    all_combinations['tied'] = np.where(all_combinations['zeroes'] == all_combinations['ones'], 1, 0)
#    all_combinations['concordant'] = np.where(all_combinations['zeroes'] < all_combinations['ones'], 1, 0)
#    all_combinations['discordant'] = np.where(all_combinations['zeroes'] > all_combinations['ones'], 1, 0)
#    tied = tied + all_combinations['tied'].sum()
#    concordant = concordant + all_combinations['concordant'].sum()
#    discordant = discordant + all_combinations['discordant'].sum()
#    del all_combinations
#
#    print('Tied : ', all_combinations['tied'].sum()/len(all_combinations), '\n', \
#      'Concordant : ', all_combinations['concordant'].sum()/len(all_combinations), '\n',\
#      'Disconcordant : ', all_combinations['discordant'].sum()/len(all_combinations), '\n', \
#      'Gini : ', all_combinations['concordant'].sum()/len(all_combinations) - all_combinations['discordant'].sum()/len(all_combinations))



# ------------------------ Random Forest --------------------------------------------------------
rf_cc_train_data = cc_temp_train.copy()
rf_cc_val_data = cc_temp_val.copy()

rf_cc_train_data['train_or_val'] = 'train'
rf_cc_val_data['train_or_val'] = 'val'

rf_cc_data = rf_cc_train_data.append(rf_cc_val_data)
rf_cc_data = rf_cc_data.drop(['age_le_30', 'age_gt_40_lt_65', 'age_gt_30_lt_65', \
                              'channel_used_to_join_grp1', 'channel_used_to_join_grp2', \
                              'cust_rel_in_beg_A', 'cust_seniority_le_60', 'cust_seniority_ge_120', \
                              'cust_seniority_ge_180', 'gender_V', 'income_le_100k', 'income_ge_150k', \
                              'province_code_grp1', 'province_code_grp2', 'province_code_grp3', \
                              'segmentation_eq_3', 'diff_le60', 'diff_ge120', 'diff_ge180'], axis = 1)

# creating new variable for channel used to join
rf_cc_data['channel_new'] = np.where(rf_cc_data['channel_used_to_join'].str[1:].isin(['04', '07','13','25']), rf_cc_data['channel_used_to_join'].str[1:], rf_cc_data['channel_used_to_join'].str[:2])

change_to_integer = ['new_cust_ind', 'ind_relation', 'address_type', 'province_code', \
                     'activity_ind', 'imputed_gross_income']
for var in change_to_integer:
    rf_cc_data[var + '_e'] = rf_cc_data[var].astype(int)
 
# Encoding categorical columns
vars_to_encode = ['gender', 'residence_ind', 'foreigner_ind', 'spouse_ind', 'deceased_ind',\
                  'emp_ind', 'cust_rel_in_beg', 'cust_type_in_beg', 'segmentation', 'channel_new']


for var in vars_to_encode:
    rf_cc_data[var + '_e'] = pd.Categorical.from_array(rf_cc_data[var]).codes

# gender : ['H', 'V'] = [0, 1]
# residence_index : ['N', 'S'] = [0, 1]
# address_type : ['N', 'S'] = [0, 1]
# spouse_index : ['N', 'S'] = [0, 1]
# deceased_index : ['N', 'S'] = [0, 1]
#cust_rel_in_beg_raw ['A' 'I' 'P' 'R' 'N'] [0 1 3 4 2]
#emp_index_raw ['N' 'A' 'B' 'F' 'S'] [3 0 1 2 4]
#cust_type_in_beg_raw ['1' '4' '3' '5' '2'] [0 3 2 4 1]
#segmentation_raw ['02' '03' '01'] [1 2 0]

# split dataset into training and validation set
rf_cc_train_data, rf_cc_val_data = rf_cc_data[rf_cc_data['train_or_val'] == 'train'],rf_cc_data[rf_cc_data['train_or_val'] == 'val']

rf_shortlisted_variables = ['activity_ind_e',   'channel_new_e', \
                                'cust_rel_in_beg_e',  'cust_type_in_beg_e', \
                                'deceased_ind_e', 'foreigner_ind_e', 'gender_e', \
                                 'ind_relation_e', 'new_cust_ind_e', \
                                 	  'segmentation_e', 'diff_maintdt_firstcontractdt', \
                                'spouse_ind_e', 'province_code_e', 'cust_seniority', 'age', 'imputed_gross_income_e',\
                                'lag_cc_1', 'lag_cc_2', 'lag_cc_3', 'lag_cc_4', 'lag_cc_5', 'lag_cc_6']

 
rf_shortlisted_variables = ['lag_cc_1', 'lag_cc_2', 'lag_cc_3', 'lag_cc_4', 'lag_cc_5', 'activity_ind_e', \
                            'lag_cc_6', 'cust_rel_in_beg_e', 'channel_new_e', 'age', 'diff_maintdt_firstcontractdt'] 
 #'emp_index_e', 'address_type_e', 'residence_index_e',
# calculate VIF between variables
a = rf_cc_train_data[rf_shortlisted_variables].copy()
for i in range(len(rf_shortlisted_variables)):
    print(rf_shortlisted_variables[i],variance_inflation_factor(a[rf_shortlisted_variables].values, a.columns.get_loc(rf_shortlisted_variables[i])))

#--------------- Creating training and validation set ----------_-------                              
x_rf_train = rf_cc_train_data[rf_shortlisted_variables]
y_rf_train = rf_cc_train_data['credit_card']                                   
                                   
x_rf_validate = rf_cc_val_data[rf_shortlisted_variables]
y_rf_validate = rf_cc_val_data['credit_card']                           
                            

# Running Grid search for Random Forest
rf_first_model = RandomForestClassifier(random_state = 1)
rf_params = [{
              'n_estimators' : [500, 750, 1000],
              'max_depth' : [8, 10, 15],
              'max_features' :  [3, 4, 5, 6, 7],
              'min_samples_leaf' : [150, 200, 250]
              }]
             
best_rf_model = sklearn.grid_search.GridSearchCV(rf_first_model, rf_params, scoring = 'accuracy', cv= 3, refit = True, verbose = 2, n_jobs = 4)
best_rf_model.fit(x_rf_train, y_rf_train) 



                            
# ------- Running Random Forest classifier ---------------------------------------
                          
rf = RandomForestClassifier(n_estimators = 500, max_depth = 10, min_samples_leaf = 150, max_features = 5, verbose = 1, class_weight = 'balanced')
rf.fit(x_rf_train, y_rf_train)
rf_predictions = rf.predict(x_rf_validate)

fpr_rf, tpr_rf, thresholds_logit = metrics.roc_curve(y_rf_validate, rf_predictions)
roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)
print(roc_auc_rf)
#0.973856890929
#0.973882682487


plt.title('Receiver Operating Characteristic for Random Forest')
plt.plot(fpr_rf, tpr_rf, 'b', label = 'AUC = %0.2f' %roc_auc_rf)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


print(2*roc_auc_rf-1)
#0.947713781859
#0.947765364973

class_combination_rf = pd.DataFrame(y_rf_validate.values, rf_predictions).reset_index().rename(columns = {'index' : 'rf_predictions', 0 : 'y_validate'}).groupby(['rf_predictions', 'y_validate']).size().reset_index().rename(columns = {0 : 'Frequency'})
print(class_combination_rf)
pd.pivot_table(class_combination_rf, values = 'Frequency', index = 'rf_predictions', columns = 'y_validate')
# TPR : 96.19
# FPR : 22.25%

#-- with lag variables
#y_validate           0      1
#rf_predictions               
#0               882743   1281
#1                13888  33541
# TPR: 96.32%
# FPR: 1.55%
#-- keeping only most important variables
#y_validate           0      1
#rf_predictions               
#0               882806   1282
#1                13825  33540



#getting variable importance for rf1
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis = 0)
indices = np.argsort(importances)[::-1]  #check with dutta
print('feature ranking:')
for f in range(x_rf_train.shape[1]):
    print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))

for index in indices:
    print(rf_shortlisted_variables[index], importances[index])

#cust_rel_in_beg_e 0.327878170375
#activity_index_e 0.280957337477
#channel_new_e 0.143494609306
#age 0.0956839238411
#cust_seniority 0.0774914671549
#segmentation_e 0.0470271867728
#new_cust_ind_e 0.00850502599626
#imputed_gross_income_e 0.00846894222876
#province_code_e 0.00435139742228
#ind_relation_e 0.00247668270009
#gender_e 0.00217840008822
#deceased_index_e 0.000872408212461
#foreigner_index_e 0.000531796760288
#cust_type_in_beg_e 8.23591277596e-05
#spouse_index_e 2.92537365831e-07

#plot feature importances of the forest
plt.figure()   
plt.title('feature importances')
plt.bar(range(x_rf_train.shape[1]), importances[indices], color = 'b', yerr = std[indices], align = 'center' )
plt.xticks(range(x_rf_train.shape[1]), indices)
plt.xlim([-1, x_rf_train.shape[1]])
plt.show()


#---------------------Predicting probabilities of zeroes and ones----------
rf_prob_predictions = rf.predict_proba(x_rf_validate)

rf_actual_and_probabilities = pd.DataFrame(rf_prob_predictions, y_rf_validate.values).reset_index().rename(columns = {'index' : 'y_validate', 0 : 'zero_probability', 1 : 'one_probability'})

# ---- ks statistics -------------
rf_prob_predictions = rf.predict_proba(x_rf_validate)
rf_actual_and_probabilities = pd.DataFrame(rf_prob_predictions, y_rf_validate.values).reset_index().rename(columns = {'index' : 'y_validate', 0 : 'zero_probability', 1 : 'one_probability'})
rf_ks_stat_data = rf_actual_and_probabilities[['y_validate', 'one_probability']]
rf_ks_stat_data = rf_ks_stat_data.sort('one_probability', ascending = False)
rf_ks_stat_data ['deciles'] = pd.qcut(rf_ks_stat_data['one_probability'].rank(method = 'first'), 10, labels = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
number_of_responders = rf_ks_stat_data.groupby('deciles')['y_validate'].sum()
total_pop = rf_ks_stat_data.groupby('deciles')['y_validate'].count()


rf_prob_predictions = rf.predict_proba(x_rf_train)
rf_actual_and_probabilities = pd.DataFrame(rf_prob_predictions, y_rf_train.values).reset_index().rename(columns = {'index' : 'y_validate', 0 : 'zero_probability', 1 : 'one_probability'})
rf_ks_stat_data = rf_actual_and_probabilities[['y_validate', 'one_probability']]
rf_ks_stat_data = rf_ks_stat_data.sort('one_probability', ascending = False)
rf_ks_stat_data ['deciles'] = pd.qcut(rf_ks_stat_data['one_probability'].rank(method = 'first'), 10, labels = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
number_of_responders = rf_ks_stat_data.groupby('deciles')['y_validate'].sum()
total_pop = rf_ks_stat_data.groupby('deciles')['y_validate'].count()






#------------- RUNNING GRID SEARCH AND COMPARING MODELS WITH BEST ITERATION VALUES ---------------

#---- running grid search -----
logistic_first_model = linear_model.LogisticRegression(random_state = 1)
logistic_params = [{
              'C' : [0.2, 0.4, 0.5, 0.8, 1],
              'class_weight' : ['balanced', {0 : 0.1, 1 : 0.9}, {0 : 0.2, 1 : 0.8}, {0 : 0.25, 1 : 0.75}],
              'max_iter' :  [100, 150, 200, 250, 300]  
              }]
             
best_logistic_model = sklearn.grid_search.GridSearchCV(logistic_first_model, logistic_params, scoring = 'accuracy', cv= 5, refit = True, verbose = 2, n_jobs = 4)
best_logistic_model.fit(cc_x_train, cc_y_train) 

logit_predictions = best_logistic_model.predict(cc_x_val)

fpr_rf, tpr_rf, thresholds_logit = metrics.roc_curve(cc_y_val, logit_predictions)
roc_auc_rf = metrics.auc(fpr_rf, fpr_rf)
print(roc_auc_rf)

class_combination_logit = pd.DataFrame(y_rf_validate.values, logit_predictions).reset_index().rename(columns = {'index' : 'logit_predictions', 0 : 'y_validate'}).groupby(['logit_predictions', 'y_validate']).size().reset_index().rename(columns = {0 : 'Frequency'})
print(class_combination_logit)
pd.pivot_table(class_combination_logit, values = 'Frequency', index = 'logit_predictions', columns = 'y_validate')
#y_validate              0      1
#logit_predictions               
#0                  885013   1787
#1                   11618  33035
# TPR : 94.87%
# FPR : 1.30%













# Running Grid search for Random Forest
rf_first_model = RandomForestClassifier(random_state = 1)
rf_params = [{
              'n_estimators' : [500, 750, 1000],
              'max_depth' : [8, 10, 15],
              'max_features' :  [3, 4, 5, 6, 7],
              'min_samples_leaf' : [150, 200, 250]
              }]
             
best_rf_model = sklearn.grid_search.GridSearchCV(rf_first_model, rf_params, scoring = 'accuracy', cv= 3, refit = True, verbose = 2, n_jobs = 4)
best_rf_model.fit(x_rf_train, y_rf_train) 


rf_predictions = best_rf_model.predict(x_rf_validate)

fpr_rf, tpr_rf, thresholds_logit = metrics.roc_curve(y_rf_validate, rf_predictions)
roc_auc_rf = metrics.auc(fpr_rf, fpr_rf)
print(roc_auc_rf)

class_combination_rf = pd.DataFrame(y_rf_validate.values, rf_predictions).reset_index().rename(columns = {'index' : 'rf_predictions', 0 : 'y_validate'}).groupby(['rf_predictions', 'y_validate']).size().reset_index().rename(columns = {0 : 'Frequency'})
print(class_combination_rf)
pd.pivot_table(class_combination_rf, values = 'Frequency', index = 'rf_predictions', columns = 'y_validate')

# tpr : 96.28%
# FPR : 1.55%







# Running grid search for GBM
gbm_first_model = GradientBoostingClassifier(random_state = 1)
gbm_params = [{
               'learning_rate' : [0.1, 0.01, 0.001],
              'n_estimators' : [250, 500, 750, 1000],
              'max_depth' : [1, 2, 3],
              'max_features' :  [3, 4, 5, 6, 7],
              'min_samples_leaf' : [50, 100, 150]
              }]
             
best_gbm_model = sklearn.grid_search.GridSearchCV(gbm_first_model, gbm_params, scoring = 'accuracy', cv= 3, refit = True, verbose = 5, n_jobs = 6)
best_gbm_model.fit(x_rf_train, y_rf_train) 

gbm_predictions = best_gbm_model.predict(x_rf_validate)

fpr_rf, tpr_rf, thresholds_logit = metrics.roc_curve(y_rf_validate, gbm_predictions)
roc_auc_rf = metrics.auc(fpr_rf, fpr_rf)
print(roc_auc_rf)

class_combination_gbm = pd.DataFrame(y_rf_validate.values, gbm_predictions).reset_index().rename(columns = {'index' : 'gbm_predictions', 0 : 'y_validate'}).groupby(['gbm_predictions', 'y_validate']).size().reset_index().rename(columns = {0 : 'Frequency'})
print(class_combination_gbm)
pd.pivot_table(class_combination_gbm, values = 'Frequency', index = 'gbm_predictions', columns = 'y_validate')
#y_validate            0      1
#gbm_predictions               
#0                882988   1308
#1                 13643  33514
# TPR : 96.24%
# FPR : 1.52%

