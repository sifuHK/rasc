# -*- coding: utf-8 -*-

from rascpy.StepwiseRegressionSKLearn import LogisticReg,LinearReg
from sklearn.datasets import make_classification,make_regression
import pandas as pd

def get_X_y(target_type,n_samples=2000,random_state=0):
    if target_type == 'logistic':
        '''
        Generate data for a binary classification problem.
        A total of 10 independent variables are generated. Variables 1-4 are informative. Variables 5 and 6 are redundant with the first four. Variables 7-10 are uninformative.
        class_sep specifies the degree of separation between the two classes. A higher value indicates easier classification.
        '''
        X, y = make_classification(n_samples=n_samples,n_features=10,n_informative=4,n_redundant=2,shuffle=False,random_state=random_state,class_sep=1)
        '''
        Convert the generated X into DataFrame (StepwiseRegressionSKLearn only supports X of DataFrame type) and name them according to the feature contribution of the independent variables
        '''
        X = pd.DataFrame(X,columns=['informative_1','informative_2','informative_3','informative_4','redundant_1','redundant_2','useless_1','useless_2','useless_3','useless_4']).sample(frac=1)
        '''
        Convert the generated y into a Series (StepwiseRegressionSKLearn only supports y of type Series)
        '''
        y=pd.Series(y).loc[X.index]
        
    if target_type == 'linear':
        '''
        Generate data for the regression problem
        Generate 10 independent variables. Variables 1-6 are informative, with a rank of approximately 4, indicating a correlation between the six variables. Variables 7-10 are uninformative.
        Noise: Add appropriate noise.
        '''
        X, y = make_regression(n_samples=n_samples,n_features=10,n_informative=6,effective_rank=4,noise=5,shuffle=False,random_state=random_state)
        '''
        Convert the generated X into a DataFrame (StepwiseRegressionSKLearn only supports X of DataFrame type) and name them according to the feature contribution of the independent variables.
        '''
        X = pd.DataFrame(X,columns=['informative_1','informative_2','informative_3','informative_4','informative_5','informative_6','useless_1','useless_2','useless_3','useless_4']).sample(frac=1)
        '''
        Convert the generated y into a Series (StepwiseRegressionSKLearn only supports y of type Series)
        '''
        y=pd.Series(y).loc[X.index]
    return X, y
    
def test_logit(X,y):
    # Output the detailed modeling process and results to logit.xlsx
    lr  =  LogisticReg(measure='aic',results_save='logit.xlsx')
    model = lr.fit(X,y)
    return model

   
def test_linear(X,y):
    # Specifying a pvalue threshold
    model  =  LinearReg(pvalue_max=0.01)
    model = model.fit(X,y)
    return model

if __name__ == '__main__':    
    X_logit, y_logit = get_X_y('logistic')
    clf  = test_logit(X_logit,y_logit)
    '''
    output
    Round 1: round increase variable informative_2
    Round 1: Current model performance :aic = 2501.870672962147
    Round 1: current input variable :['informative_2']
    Round 1 complete.Number of current input variables:1
    Round 2: round increase variable informative_3
    Round 2: Current model performance :aic = 2447.3475021471504
    Round 2: current input variable :['informative_2', 'informative_3']
    Round 2 complete.Number of current input variables:2
    Round 3: round increase variable informative_1
    Round 3: Current model performance :aic = 2442.898075769667
    Round 3: current input variable :['informative_2', 'informative_3', 'informative_1']
    Round 3 complete.Number of current input variables:3
    Round 4: under the premise of meeting the conditions set by the user, the index of the model cannot be further improved by adding or deleting variables, the modeling ends.The count of variables in model:3. 
    '''
    print(clf.predict_proba(X_logit))
    
    X_linear, y_linear = get_X_y('linear',5000)
    lin= test_linear(X_linear,y_linear)
    '''
    output
    Round 1: round increase variable informative_6
    Round 1: Current model performance :r2 = 0.0264
    Round 1: current input variable :['informative_6']
    Round 1 complete.Number of current input variables:1
    Round 2: round increase variable informative_1
    Round 2: Current model performance :r2 = 0.0403
    Round 2: current input variable :['informative_6', 'informative_1']
    Round 2 complete.Number of current input variables:2
    Round 3: round increase variable informative_2
    Round 3: Current model performance :r2 = 0.0576
    Round 3: current input variable :['informative_6', 'informative_1', 'informative_2']
    Round 3 complete.Number of current input variables:3
    Round 4: round increase variable informative_3
    Round 4: Current model performance :r2 = 0.0664
    Round 4: current input variable :['informative_6', 'informative_1', 'informative_2', 'informative_3']
    Round 4 complete.Number of current input variables:4
    Round 5: under the premise of meeting the conditions set by the user, the index of the model cannot be further improved by adding or deleting variables, the modeling ends.The count of variables in model:4. 
    '''
    print(lin.predict(X_linear))
