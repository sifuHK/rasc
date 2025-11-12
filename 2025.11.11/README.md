# rascpy
- The rascpy project has supplemented and improved existing statistical methods in order to provide data analysts and modelers with a more accurate and convenient algorithmic framework.  
- One major application area is credit risk management. rascpy provides statistical algorithms that are more accurate and easier to use than existing libraries, such as risk scorecard, bidirectional (linear/logistic) stepwise regression, optimal binning, automatic parameter search for ensemble trees, the Impute algorithm for filling and transforming complex data with both missing and special values, high-dimensional stratified sampling, and rejection inference.  
**Project evolution process:**
- Phase 1 (ScoreConflow): Provides a "business instruction + unattended" scoring card model development method. The original intention of the rascpy project is to provide more accurate algorithms and more labor-saving scoring card development tools.  
- Phase 2 (Risk Actuarial Score Card): Based on the previous phase, a risk actuarial score card is provided to build a model by comprehensively considering user default, user profit level, and data cost.
- Phase 3 (Risk Actuarial Statistics): This phase is no longer limited to scorecard development. It aims to improve the areas where statistics and machine learning fall short in actual risk measurement. Building on the previous phase, it adds features such as XGB automatic parameter search, high-dimensional stratified sampling, the Impute algorithm for filling and transforming data with both null and special values, and rejection inference.
## resource
[English Documents,Tutorials and Examples (Version = 2025.11.11)](https://github.com/sifuHK/rasc/tree/main/2025.11.11)  
[Chinese Documents, Tutorials, and Examples (version = 2025.11.11)](https://gitee.com/sifuHK/rasc/tree/master/2025.11.11)
## Install
pip install rascpy
## Version = 2025.11.11  Major Updates
1. StepwiseRegressionSKLearn has replaced Reg_Step_Wise_MP as the built-in bidirectional stepwise regression model algorithm in rascpy. StepwiseRegressionSKLearn removes a step from Reg_Step_Wise_MP, saving significant computation time. This step has a very small probability of slightly improving model performance, but consumes a significant amount of runtime. After evaluation, rascpy has decided to remove this step from StepwiseRegressionSKLearn to save computation time. Although Reg_Step_Wise_MP remains, StepwiseRegressionSKLearn is recommended.
1. The bidirectional stepwise regression model algorithm StepwiseRegressionSKLearn complies with the interface specification of sklearn and can be used in pipeline.
1. The operating language automatically switches between Chinese and English.
1. Use Excel pivot tables as a template to beautify all automatically generated reports.
1. Add support for Python 3.14
## Project Introduction
Its main functions include:
1. Provide binning algorithms that support monotonic or U-shaped constraints.
1. In addition to user-specified constraints, rascpy can automatically identify constraints (monotonic or U-shaped) that apply to the data based on the training and validation sets. Users can export the identified results to Excel and compare them with their own understanding of the variable trends.
1. Provides a more accurate binning algorithm, resulting in a higher IV. Mathematically, this set of nodes is proven to be the global optimal node, regardless of constraints or even for categorical variables.
1. This binning algorithm can also handle complex data types where a feature contains multiple special values and null values at the same time.
1. Provides Python's bidirectional stepwise regression algorithm (including logistic regression and linear regression) and supports various constraints, such as coefficient sign, p-value, and the number of group variables to be included in the model. The entire model iteration process and the reasons for each variable not being included in the model are all exported to Excel.
1. Provides a bidirectional stepwise regression algorithm with actuarial capabilities. This algorithm uses company profits as part of a loss function that considers the model's prediction accuracy, individual user profitability, and data costs. (This is during the testing phase and will undergo significant changes.)
1. Provides a more convenient missing value filling function. Common missing value filling methods can only handle missing values, but cannot handle special values, especially when the data contains both missing and special values. Special values cannot be simply equated with missing values. Simply treating special values as missing values without considering the business scenario will lead to information loss. Special values transform numeric data into a complex data type that mixes categorical and numerical data. Currently, no model can directly handle this data (although some models can produce results, they are not accurate and have no practical significance). The Impute algorithm provided by rascpy can solve this problem. The transformed data can be directly fed into any model and meet practical business requirements.
1. Provides high-precision, high-dimensional stratified sampling. The built-in stratified sampling in machine learning is overly simplistic. When stratified sampling is performed solely based on the Y label, a phenomenon may occur: after the variables in the training and test sets are segmented with equal frequency according to the same nodes, the event rates of Y differ significantly. This makes it difficult to narrow the metric differences between the training and validation sets during modeling. Without high-precision, high-dimensional stratified sampling, this problem can only be addressed by reducing model performance to improve model generalization. Another phenomenon is that after binning, the binning results of the training and validation sets differ significantly, manifesting as significant differences in IV values. Without high-precision, high-dimensional stratified sampling, the only way to improve generalization is to increase the bin width. The stratified sampling method provided by rascpy has been tested and shown to significantly mitigate this phenomenon, reducing the discrepancy between the training and validation sets without compromising model performance or binning IV. (Excessive disparity between datasets is often caused by inconsistencies in the high-dimensional joint distribution, but due to sampling precision limitations, this can only be treated as overfitting.)
1. Provides automatic parameter tuning for xgboost. Testing has shown that models created with other parameter tuning frameworks often exhibit significant discrepancies between training and validation set metrics. However, the xgboost automatic parameter tuning framework provided by rascpy minimizes the discrepancy between training and validation set metrics.
1. Support the rejection inference model of the scorecard.
1. Support the rejection inference model of xgboost.
1. Provide model report output function, users can easily generate Excel documents for model development reports.
1. Provides batch beautification and export of dataframes to Excel. The output format is similar to Excel's pivot table and has color scales.
1. Support for automated modeling. The functions provided above can be called using traditional API methods, allowing users to assemble each module through programming to build models. Alternatively, unattended modeling can be achieved by using the AI instruction templates provided by rascpy.
## Introduction to main modules
### 1.Bins
The optimal split point calculated by Bins is a set of split points that maximizes IV with a mathematical proof.
For categorical variables, including ordered and unordered categories, there is also a set of split points that can be mathematically proven to maximize IV.
Its main functions are:
1. Find the split point that maximizes IV with or without constraints. Five constraint settings are supported: monotonic (automatically determines increasing or decreasing), monotonically decreasing, monotonically increasing, U-shaped (automatically determines convex or concave), and automatically set appropriate constraints (automatically determines monotonically decreasing, monotonically increasing, convex U-shaped, and concave U-shaped).
1. For categorical variables with or without constraints, the global optimal split point can also be found to maximize IV.
1. Use "Minimum difference in event rates between adjacent bins" instead of "Information Gain" or "Chi-Square Gain" to prevent the formation of bins with too small differences. This allows users to intuitively understand the size of the differences between bins. This feature is also supported for categorical variables.
1. Do not replace the minimum value of the first bin with negative infinity, nor the maximum value of the last bin with positive infinity. This ensures that outliers are not masked by extending extreme values to infinity. RASC also provides a comprehensive mechanism to handle online values exceeding modeling boundaries. This resolves the common contradiction between the need to detect outliers as early as possible during data analysis and the need to mask them in online applications to prevent process bottlenecks (while still providing timely alerts).
1. The concept of wildcards is introduced to solve the problem that the online values of categorical variables exceed the modeling value range.
1. Support multi-process parallel computing.
1. Support binning of weighted samples.
1. Support left closed and right open binning.
  
In most cases, users do not need to interact directly with Bins components. However, rascpy is designed to be pluggable, so advanced users can use Bins modules independently, just like any other Python module.
### 2.StepwiseRegressionSKLearn
It is a linear/logistic two-way stepwise regression implemented in Python, which adds the following features to the traditional two-way stepwise regression:
1. When performing stepwise variable selection for logistic regression, AUC, KS, and LIFT metrics can be used instead of AIC and BIC. For some business scenarios, AUC and KS are more appropriate. For example, in ranking tasks, a model built using the KS metric uses fewer variables while maintaining the same KS, thereby reducing data costs.
1. When performing stepwise variable selection, use other datasets to calculate model evaluation metrics rather than the modeling dataset. Especially when the data size is large and a validation set is included in addition to the training and test sets, it is recommended to use the validation set to calculate evaluation metrics to guide variable selection. This helps reduce overfitting.
1. Supports using partial data to calculate model evaluation metrics to guide variable selection. For example, if a business requires a certain pass rate of N%, then the bad event rate can be minimized for the top N% of samples, without requiring all samples to be included in the calculation. Actual testing shows that in appropriate scenarios, using partial data as evaluation metrics results in fewer variables than using full data, but the metrics of interest to users remain unchanged. Because the model focuses only on the top, more easily distinguishable sample points, business objectives can be achieved without requiring too many variables.
1. Supports setting multiple conditions. Variables must meet all conditions simultaneously to be included in the model. Built-in conditions include: P-Value, VIF, correlation coefficient, coefficient sign, number of variables in a group, etc.
1. Supports specifying variables that must be entered into the model. If the specified variables conflict with the four conditions, a comprehensive mechanism has been designed to resolve the problem.
1. The modeling process is exported to Excel, recording the reasons for deleting each variable and the process information of each round of stepwise regression.
1. Support actuarial calculations, using company profits as a loss function that takes into account the model's prediction accuracy, the profit level of a single user, and data costs (in the testing phase, there will be significant changes later)
1. Support sklearn interface and can be used in pipeline

In most cases, users do not need to interact directly with the StepwiseRegressionSKLearn component. However, rascpy is designed to be pluggable, so advanced users can use the StepwiseRegressionSKLearn module independently, just like any other Python module.
### 3.Cutter
Perform equal frequency segmentation or segmentation according to specified split points, which has the following enhancements over the built-in segmenters of Python or Pandas:
1. A mathematically provable analytical solution with minimum global error.
1. All split points are derived from the original data. The minimum and maximum values for each interval are derived from the original data. This is different from Python or Pandas built-in splitters, which modify the minimum and maximum values at each end of each group.
1. More humane support for left closed and right open: the last group is right closed.
1. A globally optimal segmentation solution can be given even for extremely skewed data.
1. Support weighted series.
1. Supports user-specified special values. Special values are grouped separately, and users can also combine multiple special values into one group through configuration.
1. Users can specify how to handle None values. If not specified and the sequence contains null values, the null values will be automatically grouped together.
1. When a sequence is split using a specified split point, if the maximum or minimum value of the sequence exceeds the maximum or minimum value of the split point, the maximum and minimum values of the split point will be automatically extended.

It is recommended to try using Cutter to replace the built-in equal frequency segmentation component of Python or Pandas.
### 4. Other modules
There are also other modules that can significantly improve the accuracy and efficiency of data analysis and modeling:
The rascpy.Impute package can handle data with multiple special values and null values (binary classification tasks). This solves the current problem of using Impute to treat special values as None or as normal values, which can result in information loss or render the model meaningless.
1. Provides high-precision, high-dimensional stratified sampling. This solves the current problem of reducing the discrepancy between training and test set metrics by compromising model performance due to low sampling precision. rascpy.Sampling can reduce the discrepancy between training and test set metrics by minimizing the differences in dataset distribution without compromising model performance.
1. Provides automatic parameter tuning for xgboost. rascpy.Tree.auto_xgb differs from other automatic parameter tuning frameworks in that it can reduce the model variance while maintaining high training set metrics.
1. Support scorecard and xgboost rejection inference.
1. In addition to manually calling the above modules, users can choose to use AI instructions to automatically complete modeling without supervision.
## Usage Tutorial
### Scorecard Development Example
```Python
from rascpy.ScoreCard import CardFlow
if __name__ == '__main__': # Windows must write a main function (but not in jupyter), Linux and MacOS do not need to write a main function
    # Pass in the command file
    scf = CardFlow('./inst.txt')
    # There are 11 steps in total: 1. Read data, 2. Equal frequency binning, 3. Variable pre-filtering, 4. Monotonicity suggestion, 5. Optimal binning, 6. WOE conversion, 7. Variable filtering, 8. Modeling, 9. Generate scorecard, 10. Output model report, 11. Develop rejection inference scorecard
    scf.start(start_step=1,end_step=11)# will automatically give the score card + the score card for rejection inference
    
    # You can stop at any step, as follows:
    scf.start(start_step=1,end_step=10)#No scorecard will be developed for rejection inference
    scf.start(start_step=1,end_step=9)#No model report will be output
        
    # If the results of the run have not been modified, there is no need to run again. As shown below, steps 1-4 that have been run will be automatically loaded (will not be affected by restarting the computer)
    scf.start(start_step=5,end_step=8)
        
    # You can also omit start_step and end_step, abbreviated as:
    scf.start(1,10)
```
After each step of scf.start is completed, a lot of useful intermediate data will be retained. This data will be saved in the work_space specified in inst.txt as pkl. Users can manually load and access this data at any time. It can also be called through the CardFlow object instance. The intermediate results generated after each step is completed are:
- step1: scf.datas
- step2: scf.train_freqbins,scf.freqbins_stat
- step3: scf.fore_col_indices,scf.fore_filtered_cols
- step4: scf.mono_suggests,scf.mono_suggests_eventproba
- step5: scf.train_optbins,scf.optbins_stat
- step6: scf.woes
- step7: scf.col_indices,scf.filtered_cols,scf.filters_middle_data,scf.used_cols
- step8: scf.clf.in_vars,scf.clf.intercept_,scf.clf.coef_,scf.clf.perf,scf.clf.coef,scf.clf.step_proc,scf.clf.del_reasons
- step9: scf.card
- step11:scf.rejInfer.train_freqbins,scf.rejInfer.freqbins_stat,
scf.rejInfer.fore_col_indices,scf.rejInfer.fore_filtered_cols,
scf.rejInfer.mono_suggests,scf.rejInfer.mono_suggests_eventproba,
scf.rejInfer.train_optbins,scf.rejInfer.optbins_stat,scf.rejInfer.woes,
scf.rejInfer.col_indices,scf.rejInfer.filtered_cols,scf.rejInfer.filters_middle_data,scf.rejInfer.used_cols
scf.rejInfer.clf.in_vars,scf.rejInfer.clf.intercept_,scf.rejInfer.clf.coef_,scf.rejInferclf.perf,scf.rejInferclf.coef,scf.rejInfer.clf.step_proc,scf.rejInfer.clf.del_reasons
And store the newly synthesized dataset for rejection inference in scf.datas['rejData']['__synData']  
**load_step**
```Python
# load_step is only loading without execution. If your Python program is closed after execution and needs to be read again, there is no need to run it again. Just load the previous result. Even if the user closes the Python kernel or restarts the computer, the user can easily restore the CardFlow instance and call the intermediate data.
# load_step avoids the trouble of loading pkl to obtain intermediate data. CardFlow instance is equivalent to an intermediate data management container.
# For example: load all steps 5 and before, and then call them through scf.xx
from rascpy.ScoreCard import CardFlow
scf = CardFlow('./inst.txt')
scf.start(load_step = 5)
print(scf.datas)
print(scf.train_optbins)
```
#### Command file example
```txt
[PROJECT INST]
model_name = Test
work_space = ../ws
no_cores = -1

[DATA INST]
model_data_file_path = ../data/model
oot_data_file_path = ../data/oot
reject_data_file_path = ../data/rej
sample_weight_col = sample_weight
default_spec_value = {-1}

[BINS INST]
default_mono=L+
default_distr_min=0.02
default_rate_gain_min=0.001
default_bin_cnt_max = 8
default_spec_distr_min=${default_distr_min}
default_spec_comb_policy = A

[FILTER INST]
filters = {"big_homogeneity":0.99,"small_iv":0.02,"big_ivCoV":0.3,"big_corr":0.8,"big_psi":0.2}
filter_data_names = {"big_homogeneity":"train,test","small_iv":"train,test","big_ivCoV":"train,test","big_corr":"train","big_psi":"train,test"}

[MODEL INST]
measure_index=ks
pvalue_max=0.05
vif_max=2
corr_max=0.7
default_coef_sign = +

[CARD INST]
base_points=600
base_event_rate=0.067
pdo=80

[REPORT INST]
y_stat_group_cols = data_group
show_lift = 5,10,20

[REJECT INFER INST]
reject_train_data_name = rej
only_base_feas = True
```
#### Detailed description of all instructions
[English all_instructions_detailed_desc.txt](https://github.com/sifuHK/rasc/blob/main/2025.11.11/all_instructions_detailed_desc.txt)  
[中文全部指令详细说明.txt](https://gitee.com/sifuHK/rasc/blob/master/2025.11.11/全部指令详细说明.txt)  
### Optimal binning example
In the scorecard development example, rascpy.Bins.OptBin/OptBin_mp is automatically called through CardFlow.
Users can also manually call OptBin/OptBin_mp to build their own modeling solutions.
``` Python
# OptBin_mp is a multi-process version of OptBin
from rascpy.Bins import OptBin,OptBin_mp
# Main parameter description
# mono: Specifies the monotonicity constraint for each variable, such as: L+ is linearly increasing, U is automatically selected from positive U or negative U, and A is automatically selected from L+, L-, Uu, and Un based on the data. For the value range, see [BINS INST]:mono in "Detailed Description of All Instructions".
# default_mono: Default monotonicity constraint for variables not set in mono
# distr_min: Specify the minimum bin ratio of normal values except special values for each variable
# default_distr_min: If the variable is not configured in distr_min, the default minimum binning ratio of the normal value
# spec_value: specifies the special value of each variable. For the rules of writing special values, see [DATA INST]:spec_value in "Detailed description of all instructions".
# default_spec_value: The default special value of the variable that does not appear in spec_value. When the special value you configured does not exist in a certain variable, the special value configuration will be automatically ignored. This command is very convenient to use when there is a global unified special value in the data.
# spec_distr_min: The minimum percentage of each special value for each variable (when the type is a double-nested dict) or the minimum percentage of all special values for the variable (when the type is a single-layer dict). If the percentage of special values in a variable is too small, the special values are merged using the merging strategy specified by spec_comb_policy. The purpose of special value merging is to reduce abnormal results caused by special values with too small a percentage.
# default_spec_distr_min: If the variable is not in spec_distr_min, the default minimum proportion of special values under the variable. The value can be a dict (to configure the default minimum proportion for each special value separately) or a number (all special values use the same default proportion)
# spec_comb_policy: Specifies the merging rule for special values for each variable. When the proportion of the special value is less than the threshold specified by spec_distr_min, the merging rule is used. For the value range, see [BINS INST]:spec_comb_policy in "Detailed Description of All Instructions".
# default_spec_comb_policy: If the variable is not configured in spec_comb_policy, the default special value merging rule is used. If the variable has no special value, this parameter is automatically ignored.
# order_cate_vars: Specifies the ordered categorical variables in the data and gives the order of each category. ** represents a wildcard character; all unconfigured categories are merged into the wildcard character. Wildcards are well-suited for variables with long-tail distributions. If the order of a variable is set to None, lexicographic order is used.
# unorder_cate_vars: Specifies the unordered categorical variables in the data. Unordered categories will be sorted according to the event rate. Each variable corresponds to a float: if the proportion of the category is less than the threshold, it will be merged into the wildcard category. The corresponding value of the variable is None: no limit on the proportion (may cause large fluctuations)
# no_wild_treat: When a categorical variable does not have a wildcard and an uncovered category appears in the data set, the category is handled. For the value range, see [CATE DATA INST]: no_wild_treat in "Detailed Description of All Instructions".
# default_no_wild_treat: If there is no variable configured in no_wild_treat, the default handling method for this category will be used if an uncovered category occurs.
# cust_bins: User manually bins the variable
# cores: The number of CPU cores used by multiple processes. None: All cores int: When it is greater than 1, it specifies the number of cores to be used. When it is less than 0, it specifies the number of cores reserved for the system, that is, all cores minus the specified number of cores. When it is equal to 1, it turns off multiple processes and uses a single process, which is equivalent to calling OptBin
if __name__ == '__main__':# Windows must write the main function, Linux and MacOS do not need to write the main function
    optBins = OptBin_mp(X_dats,y_dats,mono={'x1':'L+','x2':'U'},default_mono='A',
                        distr_min={'x1':0.05},default_distr_min=0.02,default_rate_gain_min=0.001,
                        bin_cnt_max={'x2':5},default_bin_cnt_max=8,
                        spec_value={'x1':['{-999,-888}','{-1000,None}']}, default_spec_value=['{-999,-888}','{-1000}'],
                        spec_distr_min={'x1':{'{-1000,None}':0.01,'{-999,-888}':0.05},'x2':0.01},default_spec_distr_min=0.02,
                        spec_comb_policy={'x2':'F','x3':'L'},default_spec_comb_policy='A',
                        order_cate_vars={'x7':['v3','v1','v2'],'x8':['v5','**','v4'],'x9':None},
                        unorder_cate_vars={"x10":0.01,"x11":None},no_wild_treat={'x10':'H','x11':'m'},default_no_wild_treat='M',
                        cust_bins={'x4':['[1.0,4.0)','[4.0,9.0)','[9.0,9.0]','{-997}','{-999,-888}','{-1000,None}']},cores=-1)
```
### Bidirectional stepwise logistic regression example
In the scorecard development example, rascpy.StepwiseRegressionSKLearn.LogisticReg is automatically called through CardFlow.
Users can also manually call LogisticReg to build their own modeling solutions.
``` Python
from rascpy.StepwiseRegressionSKLearn import LogisticReg
# Generate data: There are 10 variables in total, of which the first 4 are useful variables, the middle 2 are redundant variables (there is collinearity with the first 4 variables), and the last 4 are useless variables. Add appropriate noise
X, y = make_classification(n_samples=10000,n_features=10,n_informative=4,n_redundant=2,shuffle=False,random_state=random_state,class_sep=2)
# Convert X to a DataFrame and modify the column names to match the variable effects. rascpy.StepwiseRegressionSKLearn.LogisticReg can only accept DataFrame as X
X = pd.DataFrame(X,columns=['informative_1','informative_2','informative_3','informative_4','redundant_1','redundant_2','useless_1','useless_2','useless_3','useless_4'])
# Convert y to Series. rascpy.StepwiseRegressionSKLearn.LogisticReg can only accept Series as y
y=pd.Series(y).loc[X.index]
# Main parameter description
# measure: A metric used to determine whether a parameter should be entered into the model. It supports aic, bic, roc_auc, ks, and other indicators.
# pvalue_max: The pvalue of all model variables cannot exceed this value. rasc has designed a complex and reasonable mechanism to ensure that the pvalue of all model variables is not greater than this value.
# vif_max: The vif of all input variables cannot exceed this value. rasc has designed a complex and reasonable mechanism to ensure that the vif of all input variables is not greater than this value.
# corr_max: The pairwise correlation coefficients of all model variables cannot exceed this value. rasc has designed a complex and reasonable mechanism to ensure that the pairwise correlation coefficients of all model variables are not greater than this value.
# iter_num: number of rounds of stepwise regression
# results_save: Output the model effect, information related to the model coefficients, reasons for deleting variables, and details of each round of stepwise regression to an Excel table
# Main attributes:
# Important attributes generated after calling fit
# lr.in_vars: all variables entered into the model
# lr.coef_: coefficients of the model
# lr.intercept_: intercept term of the model
# lr.perf: Summary of model performance
# lr.coef: Detailed summary of model coefficients
# lr.del_reasons: the reason for deletion of each deleted variable
# lr.step_proc: Details of each round of stepwise regression
if __name__ == '__main__':# Windows must write the main function, Linux and MacOS do not need to write the main function
    lr  =  LogisticReg(measure='roc_auc',pvalue_max=0.05,vif_max=3,corr_max=0.8,results_save = 'test_logit.xlsx')
    lr  = lr.fit(X,y)
    hat = lr.predict_proba(X)
    
# Other important parameters
# user_save_cols: variables that users are forced to enter into the model. A complex and reasonable mechanism is designed to handle conflicts between user_save_cols and commands such as pvalue_max, vif_max, and corr_max.
# coef_sign: dict, used to specify the coefficient sign of each variable
# default_coef_sign: When a variable is not in coef_sign, the default value of the variable symbol constraint
# cnt_in_group: Set the maximum number of variables allowed to be entered into each variable group
# default_cnt_in_group: If a variable group is not set in cnt_group, the default maximum number of variables allowed to be entered into the module
```
### XGB automatic parameter search example
``` Python
from rascpy.Tree import auto_xgb
# Parameter description
# cands_num: auto_xgb will give a score to each hyperparameter tried during automatic parameter search. The higher the score, the more recommended the model trained with the hyperparameter is. Then the scores are sorted from high to low, and the models with the top cands_num scores are returned.
# In actual use, the model with the highest score (i.e. clf_cands[0]) is the best model in most cases. However, users can still select their favorite model from the candidate models clf_cands[n] according to their preferences.
# cost_time: The running time of auto_xgb. Because the essence of parameter search is a combinatorial explosion, the goal of any algorithm is to find the most likely optimal set of hyperparameters within a limited time. Therefore, the longer cost_time is, the more likely it is to find the optimal set of hyperparameters.
# However, in actual use, the author has found that setting cost_time to 3-5 minutes has yielded the optimal model for most cases. Setting it longer generally fails to yield a higher-scoring model. If the user is dissatisfied with the model, they can try increasing cost_time, but increasing it to more than 8 minutes is not recommended and will likely be ineffective.
# If the user is not satisfied with the bias or variance of the model, the best way is not to increase cost_time, but to try using a more accurate sampling method, such as rascpy.Impute.BCSpecValImpute
# Return value description
# perf_cands: list. Metrics of all candidate models. Each metric contains three pieces of information: train_ks(train_auc), val_ks(val_auc), |train - val| (the absolute value of the difference between the training set and the validation set)
# params_cands: list. Hyperparameters of all candidate models
# clf_cands: list. All candidate models
# vars_cands: list. All candidate model input variables
# Note: The indexes of these 4 return values are relative. If the user decides to use the clf_cands[0] model, he can view the model's metrics through perf_cands[0], the model's hyperparameters through params_cands[0], and the model's input variables through vars_cands[0].
perf_cands,params_cands,clf_cands,vars_cands = auto_xgb(train_X,train_y,val_X,val_y,metric='ks',cost_time=60*5,cands_num=5)
proba_hat = clf_cands[0].predict_proba(X)[:,1]#The columns of X need to completely correspond to the columns during training. Even if a column is not entered into the model, it must be passed into the predict_proba method.
# When making predictions, you can also try to use the more convenient predict_proba
from rascpy.Tool import predict_proba
proba_hat = predict_proba(clf_cands[0],X[vars_cands[0]],decimals=4)#Only the variables to be input into the model need to be passed in, which is very convenient for online systems. And the returned proba_hat is a Series with the same row index as X.
```
### Impute Example
BCSpecValImpute can be used to handle special values and missing values in data for binary classification problems. It can handle special values and missing values for continuous, unordered categorical, and ordered categorical variables.
BCSpecValImpute can simultaneously fill in empty values and transform special values.
If the data contains both null values and special values, most models cannot handle them well (in a business-friendly way). We recommend using rascpy.Impute.BCSpecValImpute to preprocess the data before training it in a binary classification model.
``` Python
from rascpy.Impute import BCSpecValImpute
# Main parameter description
# spec_value: specifies the special value of each variable. For the rules of writing special values, see [DATA INST]:spec_value in "Detailed description of all instructions".
# default_spec_value: The default special value of the variable that does not appear in spec_value. When the special value you configured does not exist in a certain variable, the configuration of the special value will be automatically ignored. This command is very convenient to use when there is a global unified special value in the data.
# order_cate_vars: Specifies the ordered categorical variables in the data and gives the order of each category. ** represents a wildcard character; all unconfigured categories are merged into the wildcard character. Wildcards are well-suited for variables with long-tail distributions. If the order of a variable is set to None, lexicographic order is used.
# unorder_cate_vars: Specifies the unordered categorical variables in the data. Unordered categories will be sorted according to the event rate. If the value is float, if the proportion of the category is less than the threshold, it will be merged into the wildcard category. If the value is None, there is no limit on the proportion (which may cause large fluctuations)
# impute_None: Whether to fill in null values. Because some models can automatically handle null values, if you use such a model later, you can ignore null values when filling, and only need to handle special values. (Almost all models cannot directly handle data with both null values and special values)
bcsvi = BCSpecValImpute(spec_value={'x1':['{-999,-888}','{-1000,None}']
        ,'x11':['{unknow}']},default_spec_value=['{-999}','{-1000}'],
        order_cate_vars={'x8':['v5','**','v4'],'x9':None},
        unorder_cate_vars={"x10":0.01,"x11":None},impute_None=True,cores=None)
bcsvi.fit(trainX,trainy,weight=None) # weight=None can be omitted
trainX = bcsvi.transform(trainX)
# trainX = bcsvi.fit_transform(trainX,trainy)
testX = bcsvi.transform(testX)
    
#View the specific filling rules:
print(bcsvi.impute_values)
#Output format: {'x1':{-999:2,-888:1,-1000:0,None:0},'x2':{-999:1,-1000:0},'x8':{None:'D'},'x11':{'unknow':'A'}}}
#From the results, we can see that the special value -999 of the numeric variable x1 is filled with 2, and the empty value is filled with 0, etc. The special value 'unknow' of the categorical variable x11 is filled with A
#If the key corresponding to a variable name is not found in the first-level dict, it means that the variable has no special value in the training set and does not need to be filled. (However, it is necessary to avoid the situation where special values exist in other datasets)
```
### High-dimensional stratified sampling example
One of the most important evaluation criteria for the effectiveness of a high-dimensional stratified sampling algorithm is whether the joint distribution of **each** x variable and y can remain consistent in each data set after sampling.
If the data itself can be divided into multiple groups along different dimensions, then it is also required that the joint distribution of **each** x variable and y can remain consistent in each group in each data set after sampling.
rascpy provides the rascpy.Sampling.split_cls algorithm, which is designed for high-precision sampling of binary classification problems. Compared with multiple sampling algorithms, it shows good consistency in joint distribution, regardless of whether the data contains groups. This is especially true for x variables, which have good predictive power.
``` Python
from rascpy.Sampling import split_cls
# Main parameter description
# dat:dataframe dataset
# y:y column name of the label
# test_size: sampling ratio
# w: column name of weight
# groups: data grouping fields
train,test = split_cls(dat,y='y',test_size=0.3,w='weight',groups=['c1','c2'],random_state=0)
dat_train = dat.loc[train.index]
dat_test = dat.loc[test.index]
```
### Scorecard Rejection Inference Model example
There are three methods for developing scorecard rejection inference models. Users can choose any method based on their own situation.
Method 1: Complete the normal scorecard and rejection inference scorecard simultaneously. Suitable for developing scorecards from scratch
``` Python
from rascpy.ScoreCard import CardFlow
if __name__ == '__main__':# Windows must write the main function, Linux and MacOS do not need to write the main function
    # Pass in the command file
    scf = CardFlow('./inst.txt')
    scf.start(start_step=1,end_step=11)# will automatically generate standard scorecards and rejection inference scorecards
```
Method 2: Complete the standard scorecard first, then generate the rejection inference scorecard. This is suitable for those who have already generated the standard scorecard with rascpy and need to generate a rejection inference scorecard.
``` Python
from rascpy.ScoreCard import CardFlow
if __name__ == '__main__':
    # Pass in the command file
    scf = CardFlow('./inst.txt')
    scf.start(start_step=11,end_step=11)#If you have already run step 1 to step 10, you can set both start_step and end_step to 11 to generate a rejection inference scorecard.
```
Method 3: Directly call the CardRej module. This is suitable for those who have developed a scorecard using other python packages and then use rascpy to generate a rejection inference scorecard.
``` Python
from rascpy.ScoreCardRej import CardRej
if __name__ == '__main__':
    # Main parameter description
    # init_clf: unbiased logistic regression model
    # init_optbins_stat_train: Unbiased bin statistics. Format: {'x1':pd.DataFrame(columns=['bin','woe'])}
    # datas: data passed in by the user. Format example: {'rejData':{'rej':pd.DataFrame(),'otherRej':pd.DataFrame()},'ootData':{'oot1':pd.DataFrame(),'oot2':pd.DataFrame()}}
    # inst_file: Instruction file. The instructions are the same as those in the 'Scorecard Development Example'. See "Detailed Instructions for All Instructions". If datas is empty, all data files under [DATA INST]:xx_data_file_path in the inst_file file will be automatically loaded. If datas is not empty, the configuration of [DATA INST]:xx_data_file_path will be ignored.
    cr = CardRej(init_clf,init_optbins_stat_train,datas=None,inst_file='inst.txt')
    cr.start()
```
Refer to the intermediate data generated by the rejection inference in step 11 of the "Scorecard Development Example". The intermediate data is called by scf.rejInfer.xx in Method 1 and Method 2, and by cr.xx in Method 3.
### Tree Rejection Inference Model 
``` Python
from rascpy.TreeRej import auto_rej_xgb
# Main parameter description
# xx_w: weight of the corresponding dataset
# metric: two options, ks or auc
# Return value description
# not_rej_clf: non-rejection inference xgb model
# rej_clf: reject the inferred xgb model
# syn_train: synthetic data used to train the final round of rejection inference model
# syn_val: synthetic data used to validate the final round of rejection inference model
not_rej_clf,rej_clf,syn_train,syn_val = auto_rej_xgb(train_X,train_y,val_X,val_y,rej_train_X,rej_val_X,train_w=None,val_w=None,rej_train_w=None,rej_val_w=None,metric='auc')
```
###  Report Beautification example
Beautify a single DataFrame (Series) and export it to Excel.
``` python  
df1 = pd.DataFrame(np.random.randn(10, 4), columns=['A1', 'B1', 'C1', 'D1'])
df1['SCORE_BIN']=['[0,100)','[100,200)','[200,300)','[300,400)','[400,500)','[500,600)','[600,700)','[700,800)','[800,900)','[900,1000]']
r,c = bfy_df_like_excel_one(df1,'df1.xlsx',title=['DF1','any'],percent_cols=df1.columns,color_gradient_sep=True,text_lum=0,row_num=2,col_num=2)#
print(r,c)

``` 
Beautify multiple DataFrames (Series) and export them to the same Excel sheet with proper layout.
``` Python
from rascpy.Report import bfy_df_like_excel
df_grid=[]
df_rows1=[]#put a df outputted to 1 row in this list
df_rows2=[]#put a df outputted to 2 row in this list
df_rows3=[]#put a df outputted to 3 row in this list
df_rows4=[]#put a df outputted to 4 row in this list
#Note: The row here is not the concept of "row" in Excel
df_grid.append(df_rows1)
df_grid.append(df_rows2)
df_grid.append(df_rows3)
df_grid.append(df_rows4)

df1 = pd.DataFrame(np.random.randn(10, 4), columns=['A1', 'B1', 'C1', 'D1'])
df1['SCORE_BIN']=['[0,100)','[100,200)','[200,300)','[300,400)','[400,500)','[500,600)','[600,700)','[700,800)','[800,900)','[900,1000]'
#Output df1 to the first table in the first row
df_rows1.append({'df':df1[['SCORE_BIN','A1', 'B1', 'C1', 'D1']],'title':['notAnum'],'percent_cols':df1.columns,'color_gradient_sep':True})

df2 = pd.DataFrame(np.random.randn(2, 4), columns=['A2', 'B2', 'C2', 'D2'])

#Output df2 to the second table in row 1
df_rows1.append({'df':df2,'color_gradient_cols':['A2','C2'],'title':['percent for BC','gradient_color for AC'],'percent_cols':['B2','C2']})

df3 = pd.DataFrame(np.random.randn(15, 4), columns=['A3', 'B3', 'C3', 'D3'])
#Output df3 to the first table in the second row
df_rows2.append({'df':df3,'color_gradient_cols':['B3'],'title':['red_max==True'],'red_max':True})

df4 = pd.DataFrame(np.random.randn(4, 5), columns=['A4', 'B4', 'C4', 'D4', 'E4'])
#Output df4 to the second table in the second row
df_rows2.append({'df':df4,'color_gradient_sep':False,'text_lum':0.4,'title':['text_lum=0.4']})

df5 = pd.DataFrame(np.random.randn(10, 15), columns=map(lambda x:'Col_%d'%x,range(1,16)))
#Output df5 to the first table in the third row
df_rows3.append({'df':df5,'color_gradient_sep':True,'title':['not_align']})

df6 = pd.DataFrame({'A6':[1,2,3,4,None],'B6':[0.1,1.2,100.5,7.4,1]})
#Output df6 to the first table in row 4
df_rows4.append({'df':df6,'color_gradient_sep':True,'percent_cols':['A6']})

df7 = pd.DataFrame({'A7':[1,2,3,4],'B7':[0.1,1.2,100.5,7.4]})
#Output df7 to the second table in row 4
df_rows4.append({'df':df7,'not_color_gradient_cols':df7.columns,'title':['ALL not_color_gradient']})

#Two output methods, choose one according to actual needs
r,c = bfy_df_like_excel(df_grid,'pivot.xlsx',sheet_name='demo',red_max=False,row_num=4,col_num=4,row_gap=2,col_gap=2,align=True,ex_align=[2])
# or
with pd.ExcelWriter('pivot.xlsx') as writer:
    r,c = bfy_df_like_excel(df_grid,writer,sheet_name='demo',red_max=False,row_num=4,col_num=4,row_gap=2,col_gap=2)
print(r,c)
```
## Multi-language switching
1. Automatic switching: rascpy will automatically switch based on the language of the operating system
1. Manual switch: Manual switch by modifying xx/Lib/site-packages/rascpy/Lan.py
1. Currently supports Chinese and English. Users can expand their own language and switch manually
## Contact Information
Email: scoreconflow@gmail.com  
Email:scoreconflow@foxmail.com  
WeChat:SCF_04
