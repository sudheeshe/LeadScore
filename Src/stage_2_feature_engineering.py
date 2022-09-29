import pandas as pd
import numpy as np
import pickle as pkl
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_error, classification_report, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from Src.logger import AppLogger


class FeatureEngineering:

    def __init__(self):
        self.file = open('D:/Ineuron/Project_workshop/LeadScore/Logs/FeatureEngg_logs.txt','a+')
        self.logger = AppLogger()

    def drop_columns(self, data, columns, axis='columns'):

        """
        Description: This method helps in dropping the specified columns
        return: dataframe
        """
        try:
            self.logger.log(self.file,
                            f'Inside drop_columns method of stage_2 class >>> Started dropping the {columns} columns from dataset')

            temp_df = data.drop(columns=columns, axis=axis)

            self.logger.log(self.file, f'Dropping {columns} columns were successful, returning dataframe.')
            self.logger.log(self.file, 'Leaving drop_columns method of stage_2 class')


            return temp_df

        except Exception as e:
            self.logger.log(self.file, str(e))





    def merge_class(self,data, columns, filter_by= 0.05, name_to_replace= 'other'):

        """
        Description: This method helps in merging less frequent categories to a single category.
        params: data - dataset
                columns - columns to perform merging
                filter_by = acts as a threshold value the categories which have equal to or less than this threshold will undergo merging.
                name_to_replace = the new class name
        return: dataframe
        """

        try:
            self.logger.log(self.file,
                            f'Inside merge_class method of stage_2 class >>> Performing category merging of less frequent categories on columns {columns}')

            for col in columns:
                # Finding classes less than filter_by of value_counts
                filter_ = data[col].value_counts(normalize=True) > filter_by

                # saving the class names below filter_by value_count
                categories_to_replace = list(filter_[filter_ == False].index)

                self.logger.log(self.file,
                                f'Found {categories_to_replace} categories which are occurring less than or equal to {filter_by * 100}% on {col} column')

                # Replacing those classes with 'name_to_replace'
                data[col] = np.where(data[col].isin(categories_to_replace), name_to_replace, data[col])

                self.logger.log(self.file, f'Merged {categories_to_replace} into a single class named "{name_to_replace}" on column {col}')

            self.logger.log(self.file,
                            f'Merging of less frequent classes on {columns} columns were successful, returning dataframe.')
            self.logger.log(self.file, 'Leaving merge_class method of stage_2 class')


            return data


        except Exception as e:
            self.logger.log(self.file, str(e))





    def categorical_encoder(self, data):

        """
         Description: This method helps in encoding categorical variable.
         return: array
        """

        try:
            self.logger.log(self.file,
                            'Inside categorical_encoder method of stage_2 class >>> Making Column transformer for ordinal encoding on "education, campaign, previous" columns .')

            self.logger.log(self.file,
                            'Inside categorical_encoder method of stage_2 class >>> Making Column transformer for One Hot encoding on "job, marital, housing, loan, contact, month,day_of_week, poutcome" columns.')

            col_transformer = ColumnTransformer([

                                        ('ordinal_encoder', OrdinalEncoder(
                                                                categories=[['other', 'basic_4y', 'basic_6y', 'basic_9y',
                                                                             'high_school', 'university_degree',
                                                                                'professional_course'],
                                                                            ['1', '2', '3', '4', 'more_than_4'],
                                                                            ['0', '1', 'more_than_1']
                                                                            ],
                                                                dtype=np.int64),
                                                                ['education', 'campaign', 'previous']),

                                        ('one_hot_encoder', OneHotEncoder(
                                                                categories='auto',
                                                                drop='first'),
                                                                ['job', 'marital', 'housing', 'loan', 'contact', 'month',
                                                                'day_of_week', 'poutcome']),

                                                ], remainder='passthrough')

            self.logger.log(self.file, ' Successfully made column transformer with Ordinal and OneHot encoder, ready to apply on data.')


            encoder = col_transformer.fit(data)

            temp_df = encoder.fit_transform(data)


            filename = 'D:/Ineuron/Project_workshop/LeadScore/Pickle/categorical_encoder.pkl'
            pkl.dump(encoder, open(filename, 'wb'))

            self.logger.log(self.file, 'Saved the column transformer with Ordinal and OneHot encoder as categorical_encoder.pkl in Pickle folder, returning the transformer')
            self.logger.log(self.file, 'Leaving categorical_encoder method of stage_2 class')


            return temp_df

        except Exception as e:
            self.logger.log(self.file, str(e))

    def label_encoder(self, data):

        """
             Description: This method helps in Label encoding of target column.
             return: dataframe
        """

        try:
            self.logger.log(self.file,
                            'Inside Label_encoder method of stage_2 class >>> Starting the label encoding')

            encoder = LabelEncoder()
            encoder.fit(data)

            filename = 'D:/Ineuron/Project_workshop/LeadScore/Pickle/label_encoder.pkl'
            pkl.dump(encoder, open(filename, 'wb'))

            self.logger.log(self.file,
                            'Saved the label_encoder as label_encoder.pkl in Pickle folder,')

            array = encoder.fit_transform(data)

            self.logger.log(self.file,
                            f' Label encoding on target column was successful, returning array.')

            self.logger.log(self.file, 'Leaving Label_encoder method of stage_2 class')

            return array

        except Exception as e:
            self.logger.log(self.file, str(e))


    def smote_oversampling(self, X, Y):

        """
            Description: This method helps in oversampling the minority class.
            return: array
        """
        try:
            self.logger.log(self.file,
                            'Inside smote_oversampling method of stage_2 class >>> Starting the Oversampling using SMOTE')

            cc = SMOTE()
            x_cc, y_cc = cc.fit_resample(X, Y)

            self.logger.log(self.file,
                            f' Oversampled minority class successful, returning array.')

            self.logger.log(self.file, 'Leaving smote_oversampling method of stage_2 class')

            return x_cc, y_cc

        except Exception as e:
            self.logger.log(self.file, str(e))



    def model_report(self,model, x_train, y_train, x_test, y_test):

        y_pred = model.predict(x_test)
        logreg_accuracy = accuracy_score(y_pred, y_test)
        logreg_f1_score = f1_score(y_pred, y_test, average='micro')
        logreg_recall_score = recall_score(y_pred, y_test, average='micro')
        print("logreg SMOTETomek: ", logreg_accuracy)
        print('logreg F1 Score: ', logreg_f1_score)
        print('logreg recall Score: ', logreg_recall_score)
        logreg_mse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("logreg RMSE for prediction: %.4f" % logreg_mse)

        y_train_pred = model.predict(x_train)
        print("Accuracy for train data")
        print(classification_report(y_train, y_train_pred))
        print("Accuracy for test data")
        print(classification_report(y_test, y_pred))
        # X = StandardScaler().fit_transform(Xc1)
        plot_confusion_matrix(model, x_test, y_test,
                                      # display_labels=class_names,
                                      cmap=plt.cm.Blues,
                                      normalize='true')

        plot_precision_recall_curve(model, x_test, y_test)
        plot_roc_curve(model, x_test, y_test)

        return y_pred, model