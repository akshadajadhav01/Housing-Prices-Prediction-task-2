# Housing-Prices-Prediction-task-2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
%matplotlib inline
file_path="C:\\Users\\Dell\\Downloads\\housing.csv"
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
Data=pd.read_csv(file_path,names=column_names,delimiter=r"\s+")
Data.head() 
df=Data.copy()
print(df.dtypes)
print("Data_shape:{}".format(df.shape)) 
plt.figure()
sns.heatmap(df.isnull(),cbar=False,yticklabels=False)
df.describe()
cols=list(df.columns) 
for col in cols:
    plt.hist(df[col],bins=25)
    plt.title(col)
    plt.show()
 for col in cols:plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.scatter(df['DIS'],df['MEDV'],s=25)
plt.xlabel('DIS')
plt.ylabel('MEDV')
plt.subplot(1,2,2)
plt.scatter(df['RM'],df['MEDV'],s=25)
plt.xlabel('RM')
plt.ylabel('MEDV')
    plt.boxplot(df[col])
    plt.title(col)
    plt.show()
 plt.figure(figsize=(12,9))
sns.heatmap(df.corr().abs(),annot=True)
def correlation (df,threshold):
    col_corr=set() # names of correlated columns
    corr_matrix=df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    
    return col_corr

print(correlation(df.iloc[:,:-1],0.8))
final_df=df.drop('TAX',axis=1)
final_df.head()
for col in list(final_df.columns):
    sns.regplot(final_df[col],final_df["MEDV"])
    plt.show()
X=final_df.iloc[:,:-1]  ### features
y=final_df["MEDV"]   ### output variable
X.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
from sklearn.ensemble import GradientBoostingRegressor
model7=GradientBoostingRegressor()
model7.fit(X_train_scaled,y_train)
print("training set score:{:0.2f}".format(model7.score(X_train_scaled,y_train)))
print("test set score:{:0.2f}".format(model7.score(X_test_scaled,y_test)))
