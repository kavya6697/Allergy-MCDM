import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r'E:\Ph.D\3-JOURNALS\Obj_4-AllergyApplication\Allergy - APP\Allergy BPA\allergy_bpa_complete.csv')
df.set_index('A/C', inplace=True)

top_rh=(list((df.nlargest(20, ['RH']))['RH'].keys()))
top_ut=list((df.nlargest(20, ['UT']))['UT'].keys())
top_ot=list((df.nlargest(20, ['OT']))['OT'].keys())
top_rh_ut=list((df.nlargest(20, ['RH_UT']))['RH_UT'].keys())
top_rh_o=list((df.nlargest(20, ['RH_OT']))['RH_OT'].keys())
top_ut_o=list((df.nlargest(20, ['UT_OT']))['UT_OT'].keys())
top_normal=list((df.nlargest(20, ['N']))['N'].keys())

ref_obj={'RH':top_rh,
         'UT':top_ut,
          'OT':top_ot,
          'RH_UT':top_rh_ut,
          'RH_O':top_rh_o,
          'UT_O':top_ut_o}

#print(ref_obj)

with open('top_complete.txt', 'w') as f:
    print(ref_obj, file=f)