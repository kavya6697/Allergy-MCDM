import sys
import pandas as pd
import os
import numpy as np
import math
from math import isnan
import copy
import pyds
from pyds import MassFunction
from itertools import product
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pymcdm
import mcdm    
from matplotlib import pyplot 


tpr = dict()
fpr = dict()
#specifity , sensitivity
def metv(val, y_test):
    target_class=["RH","UT","OT","RH_UT","RH_OT","N","UT_OT"]
    met=[['class', 'sensitivity', 'specificity']]
    for j in range(len(target_class)):
        tn, tp, fn, fp = 0, 0, 0, 0 
        for i in range(len(val)):
            if(val[i] in target_class[j]): 
                if(y_test[i]==target_class[j]): tp+=1
                else: fp+=1
            else: 
                if(y_test[i]==target_class[j]): fn+=1
                else: tn+=1
        sen=tp/(tp+fn)
        spe=tn/(tn+fp)
        temp = [target_class[j], sen, spe]
        met.append(temp)
    return met

test = pd.read_csv(r'E:\Ph.D\3-JOURNALS\Obj_4-AllergyApplication\Allergy - APP\Allergy Data\allergy_data_test_ru.csv')
x_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]
#print(list(y_test))


df = pd.read_csv(r'E:\Ph.D\3-JOURNALS\Obj_4-AllergyApplication\Allergy - APP\Allergy BPA\allergy_bpa_single.csv')
df.set_index('A/C', inplace=True)

list_col = list(x_test.columns)
#alt_names = ["RH_UT","N","RH","UT_OT","RH_OT","OT","UT"] #must be in the order of BPA - ubtrain
#alt_names = ["OT","RH_OT","RH","UT","RH_UT","UT_OT","N"] #must be in the order of BPA - ideal
alt_names = ["RH","UT","OT","RH_UT","RH_OT","N","UT_OT"] #must be in the order of BPA - single
#alt_names = ["UT","RH_OT","UT_OT","N","RH_UT","RH","OT"] #must be in the order of BPA - complete 
 

ref_c1={'RH':['Milk(P)','L finger','pollen','housedust','cough','sneeze'],
        'UT':['wheat','prawns','parthenium','aspergilus','redrashes','itching'],
        'OT':['mushroom','brinjal','chicken','parthenium','cockroach','swelling'],
        'RH_UT':['maida','prawns','cockroach','cottondust','itching','sneeze'],
        'RH_OT':['greens','Milk(P)','cottondust','housedust','cough','wheezeBlocks'],
        'UT_OT':['yams','egg','cockroach','parthenium','itching','redrashes'],
        'N':['runningnose','headache','f_history']
       }

#print(ref_c1['RH'].index('curd')+1)


mew = []
topsis = []
saw = []
test_bpa_full = []
for row in x_test.iterrows():
    c = 0
    test_bpa_single = []
    test_bpa_single_mcdm = []
    for a in row[1]:
        if(c == len(x_test.columns)):
            break
        if(str(a)=='KR'): 
            c+=1
            continue
        else: 
            s = str(list_col[c])+" "+str(a)
            if(s in df.index):
                #print(df.loc[s])
                if(list_col[c] in ref_c1['RH']): 
                    df.loc[s]['RH']=df.loc[s]['RH']*ref_c1['RH'].index(list_col[c])
                if(list_col[c] in ref_c1['UT']): 
                    df.loc[s]['UT']=df.loc[s]['UT']*ref_c1['UT'].index(list_col[c])
                if(list_col[c] in ref_c1['OT']): 
                    df.loc[s]['OT']=df.loc[s]['OT']*ref_c1['OT'].index(list_col[c])
                if(list_col[c] in ref_c1['RH_UT']): 
                    df.loc[s]['RH_UT']=df.loc[s]['RH_UT']*ref_c1['RH_UT'].index(list_col[c])
                if(list_col[c] in ref_c1['RH_OT']): 
                    df.loc[s]['RH_OT']=df.loc[s]['RH_OT']*ref_c1['RH_OT'].index(list_col[c])
                if(list_col[c] in ref_c1['UT_OT']): 
                    df.loc[s]['UT_OT']=df.loc[s]['UT_OT']*ref_c1['UT_OT'].index(list_col[c])
                if(list_col[c] in ref_c1['N']): 
                    df.loc[s]['N']=df.loc[s]['N']*ref_c1['N'].index(list_col[c])
                df.loc[s]=(df.loc[s]-df.loc[s].min())/(df.loc[s].max()-df.loc[s].min())
                test_bpa_single.append(df.loc[s])
            else: 
                test_bpa_single.append([0, 0, 0, 0, 0, 0, 0])
            c += 1
    mew.append(dict(mcdm.rank(np.transpose(test_bpa_single), alt_names=alt_names, s_method="MEW")))
    #print(mew)
    topsis.append(dict(mcdm.rank(np.transpose(test_bpa_single), alt_names=alt_names, s_method="TOPSIS")))
    #print(topsis)
    saw.append(dict(mcdm.rank(np.transpose(test_bpa_single), alt_names=alt_names, s_method="SAW")))
    #print(saw)
    test_bpa_full.append(test_bpa_single)

acc = 0
acc_dum=0
saw_pred = [max(zip(saw[i].values(), saw[i].keys()))[1] for i in range(len(saw))]
#print(saw[0])
for i in range(len(list(y_test))):
    if(saw_pred[i] in y_test[i] or y_test[i] in saw_pred[i]): acc+=1
    if(saw_pred[i] == y_test[i]): acc_dum+=1
print("Accuracy_soft_SAW = ", acc/len(test_bpa_full)) 
print("Accuracy_hard_SAW = ", acc_dum/len(test_bpa_full))

saw_pred_prob_rh =[]
saw_pred_prob_ut =[]
saw_pred_prob_ot =[]
for i in range(len(saw)):
    saw_pred_prob_rh.append(saw[i]['RH'])
    saw_pred_prob_ut.append(saw[i]['UT'])
    saw_pred_prob_ot.append(saw[i]['OT'])
pyplot.yticks(fontsize=15, fontweight="bold")
pyplot.xticks(fontsize=15, fontweight="bold")
pyplot.xticks(np.arange(0, 21, 1.0))
pyplot.plot(saw_pred_prob_rh, color='red', marker='*', label='SAW')
pyplot.plot(saw_pred_prob_ut, color='blue', marker='*', label='MEW')
pyplot.plot(saw_pred_prob_ot, color='green', marker='*', label='TOPSIS')
pyplot.xlabel('instance id', fontsize=15, fontweight="bold")
pyplot.ylabel('Decision probabilities', fontsize=15, fontweight="bold")
#pyplot.legend()
pyplot.show()    

print(metrics.classification_report(y_test, saw_pred, alt_names))
cm = metrics.confusion_matrix(y_test, saw_pred, alt_names)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for SAW')
fig.colorbar(cax)
ax.set_xticklabels([''] + alt_names)
ax.set_yticklabels([''] + alt_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


acc = 0
acc_dum=0
mew_pred = [max(zip(mew[i].values(), mew[i].keys()))[1] for i in range(len(mew))]
#print(mew_pred)
for i in range(len(list(y_test))):
    if(mew_pred[i] in y_test[i] or y_test[i] in mew_pred[i]): acc+=1
    if(mew_pred[i] == y_test[i]): acc_dum+=1
print("Accuracy_soft_MEW = ", acc/len(test_bpa_full)) 
print("Accuracy_hard_MEW = ", acc_dum/len(test_bpa_full)) 


print(metrics.classification_report(y_test, mew_pred, alt_names))
cm = metrics.confusion_matrix(y_test, mew_pred, alt_names)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for MEW')
fig.colorbar(cax)
ax.set_xticklabels([''] + alt_names)
ax.set_yticklabels([''] + alt_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


acc = 0
acc_dum=0
topsis_pred = [max(zip(topsis[i].values(), topsis[i].keys()))[1] for i in range(len(topsis))]
#print(mew_pred)
for i in range(len(list(y_test))):
    if(topsis_pred[i] in y_test[i] or y_test[i] in topsis_pred[i]): acc+=1
    if(topsis_pred[i] == y_test[i]): acc_dum+=1
print("Accuracy_soft_TOPSIS = ", acc/len(test_bpa_full)) 
print("Accuracy_hard_TOPSIS = ", acc_dum/len(test_bpa_full))


print(metrics.classification_report(y_test, topsis_pred, alt_names))
cm = metrics.confusion_matrix(y_test, topsis_pred, alt_names)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for TOPSIS')
fig.colorbar(cax)
ax.set_xticklabels([''] + alt_names)
ax.set_yticklabels([''] + alt_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()




