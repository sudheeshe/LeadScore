import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Src.stage_1_data_reading import ReadData
from Src.stage_2_feature_engineering import FeatureEngineering
from Src.logger import AppLogger



class DataPipeline:

    def __init__(self):
        self.logger = AppLogger()
        self.file = self.file = open('D:/Ineuron/Project_workshop/LeadScore/Logs/DataPipeline_logs.txt','a+')

    def data_pipeline(self):

        try:
            """
              Description: This method helps in data manipulation like data cleaning and feature engg
              return: dataframe
            """
            self.logger.log(self.file,
                            f'Inside data_pipeline method of stage_3 class >>> Started data preprocessing ')

            #################################  Reading dataset   ###############################
            reader = ReadData()
            full_df = reader.read_data('D:/Ineuron/Project_workshop/LeadScore/Data/raw_data.csv')

            self.logger.log(self.file, 'Read raw_data.csv successfully as a dataframe.')


            ## Rename column names
            full_df.rename(columns={'emp.var.rate': 'emp_var_rate',
                                      'cons.price.idx': 'cons_price_idx',
                                      'cons.conf.idx': 'cons_conf_idx',
                                      'nr.employed': 'nr_employed'}, inplace=True)


            ## Renaming categories in the education colunn
            full_df['education'] = np.where(full_df['education'] == 'high.school', 'high_school', full_df['education'])
            full_df['education'] = np.where(full_df['education'] == 'university.degree', 'university_degree', full_df['education'])
            full_df['education'] = np.where(full_df['education'] == 'basic.9y', 'basic_9y', full_df['education'])
            full_df['education'] = np.where(full_df['education'] == 'professional.course', 'professional_course', full_df['education'])
            full_df['education'] = np.where(full_df['education'] == 'basic.4y', 'basic_4y', full_df['education'])
            full_df['education'] = np.where(full_df['education'] == 'basic.6y', 'basic_6y', full_df['education'])

            self.logger.log(self.file, 'renaming columns and categories done successfully.')

            ## Train - Test split
            train, test = train_test_split(full_df, test_size=0.3, random_state=0)

            test.to_csv('D:/Ineuron/Project_workshop/LeadScore/Data/test_dataframe.csv', index=False)

            self.logger.log(self.file, 'Successfully splited the data into train and test and saving test data @ Data/test_dataframe.csv .')


            ##############################  Applying Feature Engg  #############################

            self.logger.log(self.file, 'Starting Feature engineering.')

            obj_FE = FeatureEngineering()

            ## dropping Columns
            df_drop_cols = obj_FE.drop_columns(train, columns=['default', 'pdays', 'duration'], axis='columns')
            self.logger.log(self.file, "Dropping 'default', 'pdays', 'duration' columns from dataframe.")

            X, Y = df_drop_cols.drop(columns=['y'], axis=1), df_drop_cols['y']
            self.logger.log(self.file, "Splitting dataframe to X and Y.")

            ## merging less frequent colmns
            df_merge_1 = obj_FE.merge_class(X, columns=['job', 'education', 'month'], name_to_replace='other', filter_by=0.05)
            df_merge_2 = obj_FE.merge_class(df_merge_1, ['campaign'], name_to_replace='more_than_4')
            df_merge_3 = obj_FE.merge_class(df_merge_2, ['previous'], name_to_replace='more_than_1')

            self.logger.log(self.file, "Merging non frequent categories into one single categories on 'job', 'education', 'month',campaign', 'previous' columns.")


            ## applying cetegorical encoding
            self.logger.log(self.file, "Categorical encoding started on X.")
            df_encoding = obj_FE.categorical_encoder(df_merge_3)
            self.logger.log(self.file, "Label encoding started on Y.")
            label_encoded_df = obj_FE.label_encoder(Y)


            ##applying SMOTE on dataset
            self.logger.log(self.file, "Applying Oversampling by SMOTE on dataset.")
            x_smote, y_smote = obj_FE.smote_oversampling(df_encoding, label_encoded_df)

            ## Converting the array to pandas dataframe
            x_train = pd.DataFrame(x_smote)
            y_train = pd.DataFrame(y_smote)

            ## saving the dataframe
            x_train.to_csv('D:/Ineuron/Project_workshop/LeadScore/Data/x_train.csv', index=False)
            y_train.to_csv('D:/Ineuron/Project_workshop/LeadScore/Data/y_train.csv', index=False)

            self.logger.log(self.file, "Successfully completed Feature engineering on the dataset.")
            self.logger.log(self.file, "Saving the dataset @ Data/x_train.csv, Data/y_train.csv.")

            self.logger.log(self.file, "Returning dataframes x_train, y_train.")
            self.logger.log(self.file, 'Leaving data_pipeline method of stage_3 class')
            return x_train, y_train

        except Exception as e:
            self.logger.log(self.file, str(e))