import pandas as pd
import os
import numpy as np
import math
from math import isnan
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


######################DATA SPLITING USING STRATIFIED HOLD OUT APPROACH

data = pd.read_csv(r"E:\Ph.D\3-JOURNALS\Obj_4-AllergyApplication\Allergy - APP\Allergy Data\allergy_data.csv")
#print(data['Class'].value_counts())
listhead = list(data)
train, test = train_test_split(data,test_size=0.2)
print(train['Class'].value_counts())
print(test['Class'].value_counts())
train=np.array(train)
pd.DataFrame(train).to_csv(r"E:\Ph.D\3-JOURNALS\Obj_4-AllergyApplication\Allergy - APP\Allergy Data\allergy_data_ubtrain.csv")
test = np.array(test)
pd.DataFrame(test).to_csv(r"E:\Ph.D\3-JOURNALS\Obj_4-AllergyApplication\Allergy - APP\Allergy Data\allergy_data_test.csv")


