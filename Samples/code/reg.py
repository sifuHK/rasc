# -*- coding: utf-8 -*-

from rascpy.StepwiseRegressionSKLearn import LogisticReg,LinearReg
from sklearn.datasets import make_classification,make_regression
import pandas as pd

def get_X_y(target_type,n_samples=2000,random_state=0):
    if target_type == 'logistic':
        # 生成二分类问题的数据
        # 共生成10个自变量，其中第1-4自变量是有信息的。第5，6自变量与前4个是冗余的，7-10自变量是无信息的。
        # class_sep两个分类分散程度，数字越大越容易分类。
        X, y = make_classification(n_samples=n_samples,n_features=10,n_informative=4,n_redundant=2,shuffle=False,random_state=random_state,class_sep=1)
        # 将生成的X转换成DataFrame（StepwiseRegressionSKLearn只支持DataFrame类型的X），并按照自变量的特征贡献度命名它们。
        X = pd.DataFrame(X,columns=['informative_1','informative_2','informative_3','informative_4','redundant_1','redundant_2','useless_1','useless_2','useless_3','useless_4']).sample(frac=1)
        # 将生成的y转换成Series（StepwiseRegressionSKLearn只支持Serie类型s的y）
        y=pd.Series(y).loc[X.index]
        
    if target_type == 'linear':
        # 生成回归问题的数据
        # 共生成10个自变量，其中1-6自变量是有信息的，他们的秩大致为4，即6个变量之间存在一定相关性。而7-10自变量是无信息的。
        # noise:加入适当噪音
        X, y = make_regression(n_samples=n_samples,n_features=10,n_informative=6,effective_rank=4,noise=5,shuffle=False,random_state=random_state)
        # 将生成的X转换成DataFrame（StepwiseRegressionSKLearn只支持DataFrame类型的X），并按照自变量的特征贡献度命名它们。
        X = pd.DataFrame(X,columns=['informative_1','informative_2','informative_3','informative_4','informative_5','informative_6','useless_1','useless_2','useless_3','useless_4']).sample(frac=1)
        # 将生成的y转换成Series（StepwiseRegressionSKLearn只支持Serie类型s的y）
        y=pd.Series(y).loc[X.index]
    return X, y
    
def test_logit(X,y):
    #将详细建模过程与结果输出到logit.xlsx
    lr  =  LogisticReg(measure='aic',results_save='logit.xlsx')
    model = lr.fit(X,y)
    return model

   
def test_linear(X,y):
    #指定pvalue的阈值
    model  =  LinearReg(pvalue_max=0.01)
    model = model.fit(X,y)
    return model

if __name__ == '__main__':    
    X_logit, y_logit = get_X_y('logistic')
    clf  = test_logit(X_logit,y_logit)
    '''
    output
    第1轮：本轮增加变量informative_2
    第1轮：当前模型性能:aic = 2501.8706729621467
    第1轮：当前入模变量:['informative_2']
    第1轮完成。当前入模变量数量：1
    第2轮：本轮增加变量informative_3
    第2轮：当前模型性能:aic = 2447.347502147151
    第2轮：当前入模变量:['informative_2', 'informative_3']
    第2轮完成。当前入模变量数量：2
    第3轮：本轮增加变量informative_1
    第3轮：当前模型性能:aic = 2442.8980757696663
    第3轮：当前入模变量:['informative_2', 'informative_3', 'informative_1']
    第3轮完成。当前入模变量数量：3
    第4轮：在满足使用者所设置条件的前提下，已经不能通过增加或删除变量来进一步提升模型的指标，建模结束。入模变量数量：3
    '''
    print(clf.predict_proba(X_logit))
    
    X_linear, y_linear = get_X_y('linear',5000)
    lin= test_linear(X_linear,y_linear)
    '''
    output
    第1轮：本轮增加变量informative_6
    第1轮：当前模型性能:r2 = 0.0264
    第1轮：当前入模变量:['informative_6']
    第1轮完成。当前入模变量数量：1
    第2轮：本轮增加变量informative_1
    第2轮：当前模型性能:r2 = 0.0403
    第2轮：当前入模变量:['informative_6', 'informative_1']
    第2轮完成。当前入模变量数量：2
    第3轮：本轮增加变量informative_2
    第3轮：当前模型性能:r2 = 0.0576
    第3轮：当前入模变量:['informative_6', 'informative_1', 'informative_2']
    第3轮完成。当前入模变量数量：3
    第4轮：本轮增加变量informative_3
    第4轮：当前模型性能:r2 = 0.0664
    第4轮：当前入模变量:['informative_6', 'informative_1', 'informative_2', 'informative_3']
    第4轮完成。当前入模变量数量：4
    第5轮：在满足使用者所设置条件的前提下，已经不能通过增加或删除变量来进一步提升模型的指标，建模结束。入模变量数量：4
    '''
    print(lin.predict(X_linear))