import pickle as pkl
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from Src.stage_2_feature_engineering import FeatureEngineering


x_train = pd.read_csv('D:/Ineuron/Project_workshop/LeadScore/Data/x_train.csv')
y_train = pd.read_csv('D:/Ineuron/Project_workshop/LeadScore/Data/y_train.csv')

test = pd.read_csv('D:/Ineuron/Project_workshop/LeadScore/Data/test_dataframe.csv')


x_test = test.drop(['y', 'default', 'pdays'], axis='columns')
y_test = test['y']


x_test.rename(columns={'emp.var.rate': 'emp_var_rate',
                          'cons.price.idx': 'cons_price_idx',
                          'cons.conf.idx': 'cons_conf_idx',
                          'nr.employed': 'nr_employed'}, inplace=True)


x_test['education'] = np.where(x_test['education'] == 'high.school', 'high_school', x_test['education'])
x_test['education'] = np.where(x_test['education'] == 'university.degree', 'university_degree', x_test['education'])
x_test['education'] = np.where(x_test['education'] == 'basic.9y', 'basic_9y', x_test['education'])
x_test['education'] = np.where(x_test['education'] == 'professional.course', 'professional_course', x_test['education'])
x_test['education'] = np.where(x_test['education'] == 'basic.4y', 'basic_4y', x_test['education'])
x_test['education'] = np.where(x_test['education'] == 'basic.6y', 'basic_6y', x_test['education'])

obj_FE = FeatureEngineering()

df_merge_1 = obj_FE.merge_class(x_test, columns=['job', 'education', 'month'], name_to_replace='other', filter_by=0.05)
df_merge_2 = obj_FE.merge_class(df_merge_1, ['campaign'], name_to_replace='more_than_4')
df_merge_3 = obj_FE.merge_class(df_merge_2, ['previous'], name_to_replace='more_than_1')


cat_encoder = pkl.load(open('D:/Ineuron/Project_workshop/LeadScore/Pickle/categorical_encoder.pkl', 'rb'))
lab_encoder = pkl.load(open('D:/Ineuron/Project_workshop/LeadScore/Pickle/label_encoder.pkl', 'rb'))

x_test_final = cat_encoder.transform(df_merge_3)
y_test_final = lab_encoder.transform(y_test)

model = pkl.load(open('D:/Ineuron/Project_workshop/LeadScore/Models/Logistic_Regressor.pkl', 'rb'))

pred = model.predict(x_test_final)

#print(pred[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]])

obj_FE.model_report(model, x_train, y_train, x_test_final, y_test_final)