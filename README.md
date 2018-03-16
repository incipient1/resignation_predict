
# [DC学院\_员工离职预测训练赛](http://www.dcjingsai.com/common/cmpt/%E5%91%98%E5%B7%A5%E7%A6%BB%E8%81%8C%E9%A2%84%E6%B5%8B%E8%AE%AD%E7%BB%83%E8%B5%9B_%E7%BB%93%E6%9E%9C%E6%8F%90%E4%BA%A4.html)

## 数据探索


```python
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import numpy as np
```


```python
lr = linear_model.LogisticRegression()
```


```python
resignation = pd.read_csv('E:/MySQL_data/DataCastle/pfm_train.csv',encoding='utf-8')
```


```python
resignation.describe().columns
```




    Index(['Age', 'Attrition', 'DistanceFromHome', 'Education', 'EmployeeNumber',
           'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
           'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
           'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
           'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
           'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
           'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
           'department', 'education_field'],
          dtype='object')




```python
resignation.columns
```




    Index(['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
           'Education', 'EducationField', 'EmployeeNumber',
           'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel',
           'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
           'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike',
           'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
           'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
           'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
           'YearsSinceLastPromotion', 'YearsWithCurrManager', 'business_travel',
           'department', 'education_field', 'gender'],
          dtype='object')




```python
最多有9个feature需要标签化
```


```python
resignation.sample(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>410</th>
      <td>47</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>25</td>
      <td>3</td>
      <td>Medical</td>
      <td>1993</td>
      <td>1</td>
      <td>Male</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>25</td>
      <td>3</td>
      <td>3</td>
      <td>17</td>
      <td>14</td>
      <td>12</td>
      <td>11</td>
    </tr>
    <tr>
      <th>537</th>
      <td>53</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>Sales</td>
      <td>2</td>
      <td>2</td>
      <td>Medical</td>
      <td>981</td>
      <td>3</td>
      <td>Male</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>1</td>
      <td>22</td>
      <td>3</td>
      <td>4</td>
      <td>17</td>
      <td>13</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>252</th>
      <td>34</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>Sales</td>
      <td>1</td>
      <td>4</td>
      <td>Marketing</td>
      <td>1047</td>
      <td>2</td>
      <td>Male</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>16</td>
      <td>2</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>445</th>
      <td>36</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>9</td>
      <td>2</td>
      <td>Medical</td>
      <td>699</td>
      <td>2</td>
      <td>Male</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>1</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>43</td>
      <td>0</td>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>10</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>430</td>
      <td>3</td>
      <td>Female</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>9</td>
      <td>5</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



## 使用已有标签


```python
features = ['Age','DistanceFromHome','Education','EmployeeNumber',
            'EnvironmentSatisfaction','JobInvolvement','JobLevel',
            'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
            'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
            'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion',
            'YearsWithCurrManager']
x = resignation[features]
y = resignation['Attrition']
```


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
dt = DecisionTreeClassifier()
score_dt = cross_val_score(dt,x,y,cv=5,scoring='accuracy').mean()
score_dt
```




    0.74553709787392308




```python
from sklearn import neighbors

best_num = 1
knn_best_num = neighbors.KNeighborsClassifier(best_num,weights='distance')
best_score = cross_val_score(knn_best_num,x,y,cv=5,scoring='accuracy').mean()
for i in range(2,10):
    knn_i = neighbors.KNeighborsClassifier(i,weights='distance')
    score_i = cross_val_score(knn_i,x,y,cv=5,scoring='accuracy').mean()
    if score_i > best_score:
        best_score = score_i
        best_num = i

print('最佳近邻数量：{}，准确率{}'.format(best_num,best_score))
```

    最佳近邻数量：7，准确率0.8218194027299587
    


```python
knn_best_num = neighbors.KNeighborsClassifier(best_num,weights='distance')
```


```python
from sklearn import ensemble

best_deep = 1
rf_best_deep = ensemble.RandomForestClassifier(best_deep)
best_score = cross_val_score(rf_best_deep,x,y,cv=5,scoring='accuracy').mean()
for deep in range(2,24):
    rf_deep = ensemble.RandomForestClassifier(deep)
    score_deep = cross_val_score(rf_deep,x,y,cv=5,scoring='accuracy').mean()
    if score_deep > best_score:
        best_score = score_deep
        best_deep = deep

print('最佳深度：{}，准确率{}'.format(best_deep,best_score))
```

    最佳深度：19，准确率0.8482165108595405
    


```python
rf_best_deep = ensemble.RandomForestClassifier(best_deep)
```


```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
score_gnb = cross_val_score(gnb,x,y,cv=5,scoring='accuracy').mean()
score_gnb
```




    0.71910704390962243




```python
from sklearn.linear_model.stochastic_gradient import SGDClassifier
sgdc = SGDClassifier()
score_sgdc = cross_val_score(sgdc,x,y,cv=5,scoring='accuracy').mean()
score_sgdc
```

    E:\miniconda\envs\course_py35\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:84: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)
    




    0.8381874155927338




```python
from sklearn.linear_model.perceptron import Perceptron
pct = Perceptron()
score_pct = cross_val_score(pct,x,y,cv=5,scoring='accuracy').mean()
score_pct
```

    E:\miniconda\envs\course_py35\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:84: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.perceptron.Perceptron'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)
    




    0.82274494777315077




```python
from sklearn.svm import SVC
svc = SVC()
score_svc = cross_val_score(svc,x,y,cv=5,scoring='accuracy').mean()
score_svc
```




    0.8381874155927338




```python
from sklearn.svm import LinearSVC
lsvc = LinearSVC()
score_lsvc = cross_val_score(lsvc,x,y,cv=5,scoring='accuracy').mean()
score_lsvc
```




    0.83456750609047137




```python
models = [lsvc,svc,pct,sgdc,gnb,rf_best_deep,knn_best_num,dt,lr]
model_df = pd.DataFrame(np.zeros((len(models),2)),columns=['model','score'])

for i in range(len(models)):
    model_df.loc[i,'model'] = str(models[i]).split('(')[0]
    model_df.loc[i,'score'] = cross_val_score(models[i],x,y,cv=5,
                                              scoring='accuracy').mean() 

model_df.sort_values(by='score',ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>LogisticRegression</td>
      <td>0.852712</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RandomForestClassifier</td>
      <td>0.840006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVC</td>
      <td>0.838187</td>
    </tr>
    <tr>
      <th>0</th>
      <td>LinearSVC</td>
      <td>0.836377</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Perceptron</td>
      <td>0.822745</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KNeighborsClassifier</td>
      <td>0.821819</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DecisionTreeClassifier</td>
      <td>0.744669</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GaussianNB</td>
      <td>0.719107</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SGDClassifier</td>
      <td>0.702114</td>
    </tr>
  </tbody>
</table>
</div>



不同模型在处理同一数据时的能力不同，选择最优秀的模型进行预测


```python
model_lr = lr.fit(x,y)
```


```python
resignation_test = pd.read_csv('E:\\MySQL_data\\DataCastle\\pfm_test.csv',
                               encoding='utf-8')
```


```python
x_test = resignation_test[features]
pfm_predict = model_lr.predict(x_test)
```


```python
pd.DataFrame(pfm_predict).to_csv('E:\\MySQL_data\\DataCastle\\pfm_predict.csv',
                                index = False,
                                header = ['result'],
                                encoding = 'utf-8')
```

第1次提交到DC后得分：0.87428，排名142<br>
![第1次提交](E:/git/signation_predict_Data_Castal/score_1_dc.png)

## 标签化9列

### 标签化BusinessTravel

BusinessTravel|商务差旅频率
--------------|----------
Non-Travel    |不出差
Travel_Rarely |不经常出差
Travel_Frequently|经常出差


```python
set(resignation['BusinessTravel'])
```




    {'Non-Travel', 'Travel_Frequently', 'Travel_Rarely'}




```python
resignation['business_travel'] = resignation['BusinessTravel']
```


```python
resignation.loc[resignation['business_travel'] == 'Non-Travel','business_travel'] = 0
resignation.loc[resignation['business_travel'] == 'Travel_Rarely','business_travel'] = 1
resignation.loc[resignation['business_travel'] == 'Travel_Frequently','business_travel'] = 2  
```


```python
set(resignation['business_travel'])
```




    {0, 1, 2}



标签化成功

### 标签化Department

Department|员工所在部门
-----|-----
Sales|销售部
Research & Development|研发部
Human Resources|人力资源部

标签和部门之间没有关系，故自动标签


```python
resignation['department'] = resignation['Department']
```


```python
set(resignation['department'])
```




    {'Human Resources', 'Research & Development', 'Sales'}




```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(resignation['department'])
resignation['department'] = le.transform(resignation['department'])
```

### 标签化EducationField


```python
set(resignation['EducationField'])
```




    {'Human Resources',
     'Life Sciences',
     'Marketing',
     'Medical',
     'Other',
     'Technical Degree'}




```python
resignation['education_field'] = resignation['EducationField']
```


```python
le.fit(resignation['education_field'])
resignation['education_field'] = le.transform(resignation['education_field'])
```

### 标签化Gender


```python
resignation['gender'] = resignation['Gender']
resignation.loc[resignation['gender'] == 'Female','gender'] = 0
resignation.loc[resignation['gender'] == 'Male','gender'] = 1
```

### 标签化JobRole


```python
set(resignation['job_role'])
```




    {0, 1, 2, 3, 4, 5, 6, 7, 8}




```python
resignation['job_role'] = resignation['JobRole']
le.fit(resignation['job_role'])
resignation['job_role'] = le.transform(resignation['job_role'])
```

### 标签化MaritalStatus


```python
set(resignation['marital_status'])
```




    {0, 1, 2}




```python
resignation['marital_status'] = resignation['MaritalStatus']
le.fit(resignation['marital_status'])
resignation['marital_status'] = le.transform(resignation['marital_status'])
```

### 标签化OverTime


```python
resignation['over_time'] = resignation['OverTime']
```


```python
le.fit(resignation['over_time'])
resignation['over_time'] = le.transform(resignation['over_time'])
```

## 新标签训练


```python
new_labels = ['business_travel','department','education_field','gender','job_role',
             'marital_status','over_time']
features.extend(new_labels)
```


```python
x_wrap = resignation[list(set(features))]
```


```python
models = [lsvc,svc,pct,sgdc,gnb,rf_best_deep,knn_best_num,dt,lr]
model_df = pd.DataFrame(np.zeros((len(models),2)),columns=['model','score'])

for i in range(len(models)):
    model_df.loc[i,'model'] = str(models[i]).split('(')[0]
    model_df.loc[i,'score'] = cross_val_score(models[i],x_wrap,y,cv=5,
                                              scoring='accuracy').mean() 

model_df.sort_values(by='score',ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>LogisticRegression</td>
      <td>0.874486</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RandomForestClassifier</td>
      <td>0.854576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVC</td>
      <td>0.838187</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Perceptron</td>
      <td>0.838187</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SGDClassifier</td>
      <td>0.838187</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KNeighborsClassifier</td>
      <td>0.818183</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DecisionTreeClassifier</td>
      <td>0.783658</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GaussianNB</td>
      <td>0.754550</td>
    </tr>
    <tr>
      <th>0</th>
      <td>LinearSVC</td>
      <td>0.721499</td>
    </tr>
  </tbody>
</table>
</div>



最优模型lr提高了0.02分：从0.85提高到0.87，效果不明显

## 新标签预测

### 标签化，同上


```python
resignation_test['business_travel'] = resignation_test['BusinessTravel']
resignation_test.loc[resignation_test['business_travel'] == 
                     'Non-Travel','business_travel'] = 0
resignation_test.loc[resignation_test['business_travel'] == 
                     'Travel_Rarely','business_travel'] = 1
resignation_test.loc[resignation_test['business_travel'] == 
                     'Travel_Frequently','business_travel'] = 2  
```


```python
resignation_test['department'] = resignation_test['Department']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(resignation_test['department'])
resignation_test['department'] = le.transform(resignation_test['department'])

resignation_test['education_field'] = resignation_test['EducationField']
le.fit(resignation_test['education_field'])
resignation_test['education_field'] = le.transform(
    resignation_test['education_field'])

resignation_test['gender'] = resignation_test['Gender']
resignation_test.loc[resignation_test['gender'] == 'Female','gender'] = 0
resignation_test.loc[resignation_test['gender'] == 'Male','gender'] = 1

resignation_test['job_role'] = resignation_test['JobRole']
le.fit(resignation_test['job_role'])
resignation_test['job_role'] = le.transform(resignation_test['job_role'])

resignation_test['marital_status'] = resignation_test['MaritalStatus']
le.fit(resignation_test['marital_status'])
resignation_test['marital_status'] = le.transform(
    resignation_test['marital_status'])

resignation_test['over_time'] = resignation_test['OverTime']
le.fit(resignation_test['over_time'])
resignation_test['over_time'] = le.transform(resignation_test['over_time'])
```

### 预测


```python
x_test_wrap = resignation_test[list(set(features))]
```


```python
model_lr_wrap = lr.fit(x_wrap,y)
```


```python
y_test_wrap = model_lr_wrap.predict(x_test_wrap)
pd.DataFrame(y_test_wrap).to_csv('E:\\MySQL_data\\DataCastle\\pfm_predict_wrap.csv',
                                index = False,
                                header = ['result'],
                                encoding = 'utf-8')
```

第2次提交后得分：0.88857，提高了0.01分；排名：85，提高了57<br>
![第2次得分](E:\\git\\signation_predict_Data_Castal\\score_2_dc.png)