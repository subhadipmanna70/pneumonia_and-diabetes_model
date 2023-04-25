import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

# loading the dataset

df = pd.read_csv("./dataset/diabetes.csv")


df.head()


print(f'The dataset contains {df.shape} rows and columns respectively')


df.info()

df.describe()

# Replacing columns with lower case letters

df.columns = df.columns.str.lower()


#checking for missing data / values

df.isnull().sum()


df['bloodpressure'].value_counts


# Checking for Zero / Inapprorpiate Values

for col in df.columns:
    zero_values = len(df[df[col] <= 0])
    print(f'Zero values in column {col} = {zero_values}')


# Replacing Zero / Inapprorpiate Values with Median values

col_containing_zero = ["glucose","bloodpressure","skinthickness","insulin","bmi"]
for col in col_containing_zero:
    median = df[col].median()
    df.loc[df[col]<=0,col] = median


# Checking for Zero / Inapprorpiate Values after replacing with Median values

for col in df.columns:
    zero_values = len(df[df[col] <= 0])
    print(f'Zero values in column {col} = {zero_values}')


outcome_sum = df['outcome'].value_counts()
print(f'Patient without Diabetes = {outcome_sum[0]} & Patient with Diabetes = {outcome_sum[1]}')


diabetes_count = np.array(df['outcome'].value_counts(sort=True))

labels = ['Patient without Diabetes = 500 ', 'Patient with Diabetes = 268']

plt.figure(figsize=(8,6))
plt.pie(diabetes_count, labels=labels)
plt.legend()
plt.title('Comparing Diabetic and non-Diabetic patients ')
plt.show()

# making distplots of the following columns to see the distribution of data :

dist_columns = ['pregnancies', 'glucose',"bloodpressure","skinthickness","insulin","bmi","diabetespedigreefunction","age"]

for col in dist_columns:
    plt.figure(figsize=(8,6))
    sns.displot(df[col], kde = True)
    plt.title(f"Distribution of {col} column")
    plt.show()


corr_columns = ['pregnancies', 'glucose',"bloodpressure","skinthickness","insulin","bmi","diabetespedigreefunction"]

for col in corr_columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="age",y=col,data=df,hue="outcome")
    plt.title(f"Comparing distribution of {col} of patients with diabetes and without diabetes")
    plt.show()

### Main Machine Learning Code:

#Splittng the data as Train 20%,Validate 25% and Test data 60%

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)
df_full_train = df_full_train.reset_index(drop = True)


y_train = df_train.outcome.values
y_val   = df_val.outcome.values
y_test  = df_test.outcome.values
y_full_train = df_full_train.outcome.values


#Removed the Target Variable from the dataset

del df_train['outcome']
del df_val['outcome']
del df_test['outcome']
del df_full_train['outcome']


dt = DecisionTreeClassifier()
rfc = RandomForestClassifier()

models = [dt, rfc]
model_names = ['DecisionTreeClassifier','RandomForestClassifier']
mean_score = []

for model in models:
    cross_score = cross_val_score(model,df_train,y_train,cv=5)
    average_score = np.mean(cross_score)
    mean_score.append(average_score)

# plotting bar chart to compare the model performances and pointing out the best one

plt.figure(figsize=(8,6))
sns.barplot(x=model_names,y=mean_score, alpha=0.8)
plt.title("Comparison of Performance of Models")
plt.xticks(rotation=90)
plt.show()

###Using of Random Forest model Tuning

rf = RandomForestClassifier(n_estimators=10,random_state=1)
rf.fit(df_train,y_train)
y_pred = rf.predict_proba(df_val)[:, 1]

#check the roc_auc score for the trained model

roc_auc_score(y_val,y_pred)

# Looking for the best n_estimators

scores = []

for n in range(10, 251, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(df_train, y_train)

    y_pred = rf.predict_proba(df_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    scores.append((n, auc))

columns = ['n_estimators', 'auc_score']
df_scores = pd.DataFrame(scores, columns=columns)

plt.plot(df_scores.n_estimators,df_scores.auc_score)

# Looking for the best max_depth

scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(df_train, y_train)

        y_pred = rf.predict_proba(df_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))

columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]

    plt.plot(df_subset.n_estimators, df_subset.auc, label='max_depth = %s' % d)
    plt.legend()

max_depth = 5

# Looking for the best min_samples_leaf

scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(df_train, y_train)

        y_pred = rf.predict_proba(df_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))

columns1 = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores1 = pd.DataFrame(scores, columns=columns1)

for s in [1, 3, 5, 10, 50]:
    df_subset = df_scores1[df_scores1.min_samples_leaf == s]

    plt.plot(df_subset.n_estimators, df_subset.auc, label='min_samples_leaf = %s' % s)
    plt.legend()

rf = RandomForestClassifier(n_estimators=60,max_depth=5, min_samples_leaf=1,random_state=1)
rf.fit(df_train,y_train)
y_pred = rf.predict_proba(df_val)[:, 1]


final_roc_auc_score = roc_auc_score(y_val,y_pred)
print(f'Accuracy of Tuned Randomforest Model = {round(final_roc_auc_score, 4)}')


###### Checking the Model on Test Data

rf = RandomForestClassifier(n_estimators=60,max_depth=5, min_samples_leaf=1,random_state=1)
rf.fit(df_full_train,y_full_train)
y_pred = rf.predict_proba(df_test)[:, 1]


test_roc_auc_score = roc_auc_score(y_test,y_pred)
print(f'Accuracy on Test data = {round(test_roc_auc_score, 4)}')

pickle.dump(rf, open('rf_model.pkl', 'wb'))

pickled_model = pickle.load(open('rf_model.pkl', 'rb'))

data = np.array([[1,85,66,29,0,26.6,0.351,31]])


