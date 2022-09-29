import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from Src.logger import AppLogger
from Src.stage_3_data_pipeline import DataPipeline
import pickle as pkl

params = [{'penalty': ['l1', 'l2', 'elasticnet', 'none'],
         'C': np.linspace(0.001, 2.0, 100),
         'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
         'max_iter': [100, 500, 1000]}]

def model_tuning(x_train,y_train, params):

    lr = LogisticRegression()
    clf = RandomizedSearchCV(lr, param_distributions=params, n_iter=30, scoring='f1', cv=5, verbose=True)
    best_clf = clf.fit(x_train, y_train)
    best_train_score = best_clf.best_score_
    best_model_params = best_clf.best_params_

    return best_train_score, best_model_params


def model_training():
    try:
        logger = AppLogger()
        file = open('D:/Ineuron/Project_workshop/LeadScore/Logs/ModelTraining_logs.txt', 'a+')

        ##importing data_pipeline method
        pipeline = DataPipeline()
        x_train, y_train = pipeline.data_pipeline()


        logger.log(file, "x_train & y_train dataframes read successfully")


        model = LogisticRegression(C=0.01, max_iter=1000, penalty='none')
        model.fit(x_train,y_train)


        logger.log(file, "LogisticRegressor trained successfully")

        #filename = 'D:/Ineuron/Project_workshop/LeadScore/Models/Logistic_Regressor.pkl'
        #pkl.dump(model, open(filename, 'wb'))

        logger.log(file, "LogisticRegressor model saved successfully")

    except Exception as e:
        print(e)


if __name__ == "__main__":
        model_training()