import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn2, venn2_circles

df = pd.read_csv(r'E:\Ph.D\3-JOURNALS\Obj_4-AllergyApplication\Allergy - APP\Allergy BPA\allergy_bpa_complete.csv')
df.set_index('A/C', inplace=True)

gain_RH=[]
gain_UT=[]
gain_OT=[]
gain_RH_UT=[]
gain_RH_OT=[]
gain_UT_OT=[]
gain_N=[]

for key, val in dict(df.apply(lambda x: x.argmax(), axis=1)).items():
    locals()[str('gain_'+df.columns[val])].append(key)
    
print("Number of gain - complete:",
"RH",len(set(gain_RH)),
"UT",len(set(gain_UT)),
"OT",len(set(gain_OT)),
"NORMAL",len(set(gain_N)),
"RH_UT",len(set(gain_RH_UT)),
"RH_OT",len(set(gain_RH_OT)),
"UT_OT",len(set(gain_UT_OT)))

labels=('RH','UT','OT')
ax = plt.gca() 
full_RH=gain_RH+gain_RH_UT+gain_RH_OT
full_UT=gain_UT+gain_RH_UT+gain_UT_OT
full_OT=gain_OT+gain_RH_OT+gain_UT_OT

venn3([set(full_RH), set(full_UT), set(full_OT)], set_labels=labels, ax=ax,set_colors=    
      ('darkviolet','deepskyblue','blue'), alpha=0.7)
plt.show()