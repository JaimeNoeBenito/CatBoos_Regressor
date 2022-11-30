# Databricks notebook source
# REFERENCE where the main code was obtained
# https://www.kaggle.com/code/anzarwani2/price-prediction-using-catboost/notebook

# COMMAND ----------

# MAGIC %pip install catboost

# COMMAND ----------

# MAGIC %pip install hyperopt

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

# MAGIC %run Pricing/pdm_functions

# COMMAND ----------

dbutils.widgets.text("sub", "")
sub = dbutils.widgets.get("sub")
print("sub = ", sub)

dbutils.widgets.text("min_sale_year", "")
min_sale_year = dbutils.widgets.get("min_sale_year")
print("min_sale_year = ", min_sale_year)

dbutils.widgets.text("run_hyperparameters", "")
run_hyperparameters = dbutils.widgets.get("run_hyperparameters")
print("run_hyperparameters = ", run_hyperparameters)

# COMMAND ----------

# MAGIC %md ### MLflow Tracking
# MAGIC [MLflow tracking](https://www.mlflow.org/docs/latest/tracking.html) allows you to organize your machine learning training code, parameters, and models. 
# MAGIC 
# MAGIC You can enable automatic MLflow tracking by using [*autologging*](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging).

# COMMAND ----------

import mlflow

# COMMAND ----------

# Enable MLflow autologging for this notebook
mlflow.autolog()

# COMMAND ----------

# MAGIC %md ##Load data 


# COMMAND ----------

data_to_pred = pd.read_csv("<your_datalake_path>", sep = ';',  encoding = 'utf-8') #, dtype=str

# COMMAND ----------

df = data_to_pred

# COMMAND ----------

total_col = df.columns
for c in total_col:
    print(c)

# COMMAND ----------

cols_to_drop = ["id_col_1",
			    "id_col_2"
               ]

for cd in cols_to_drop:
    df = df.drop(cd , axis=1)

# COMMAND ----------


df.info()

# COMMAND ----------

# MAGIC %md ### Field subset manual

# COMMAND ----------

col_subset = [
                     "Make",
                     "Model",
                     "Price_bought",
                     "Cubic_engine",
                     "Mileage",
                     "Co2_Emissions", 
					 "other_fiels_relevants....",
                     "Sale_Price"
    ]


data = df[col_subset]
df = data


# COMMAND ----------

s = (df.dtypes == 'object')
object_cols = list(s[s].index)

uniqueList_objects = []
for letter in object_cols:
    if letter not in uniqueList_objects:
        uniqueList_objects.append(letter)

print(uniqueList_objects)

# COMMAND ----------

# MAGIC %md ### Normalization missing / incorrect values

# COMMAND ----------

from catboost import CatBoostRegressor

# COMMAND ----------

# MAGIC %md ### Numeric conversions 

df['Cubic_engine']=df['Cubic_engine'].fillna(0).astype('int64')
df['Price_bought']=df['Price_bought'].astype('int64')
df['Mileage']=df['Mileage'].fillna(0).astype('int64')
df['Co2_Emissions']=df['Co2_Emissions'].fillna(0).astype('int64')
df['other_fiels_relevants']=df['other_fiels_relevants'].fillna(0).astype('int64')

# COMMAND ----------

df.info()

# COMMAND ----------

categorical_features_indices = np.where((df.dtypes != np.float) & (df.dtypes != np.int64))[0]

for pos in categorical_features_indices:
#     print(pos)
    colname = df.columns[pos]
    print(colname)
    df[[colname]] = df[[colname]].fillna('')
    df[[colname]] = df[[colname]].replace(np.nan,'',regex = True)

# COMMAND ----------

# MAGIC %md ##Outliers for the sale price

# COMMAND ----------
ul = round(df["Sale_Price"].quantile(0.99),0)
print("Upper limit:   ", ul)
    
#     print("Los registros con valores atípicos graves con valores en el extremo superior son: ")
#     print(df_tmp[df_tmp["Sale_Price"]> (df_tmp["Sale_Price"].quantile(0.75) + (3 * (df_tmp["Sale_Price"].quantile(0.75)-df_tmp["Sale_Price"].quantile(0.25))))].sort_values("Sale_Price",ascending = False))
    

ll = round(df["Sale_Price"].quantile(0.01),0)
print("Lower limit:  ", ll)

# COMMAND ----------

# MAGIC %md ### Remove or asign new values to the outliers

# COMMAND ----------

df = df[df['Sale_Price'] > ll]
df = df[df['Sale_Price'] < ul]

# COMMAND ----------

# pra evitar que estos valores extremos perjudiquen al análisis, se ha decidido utilizar **winsorizacion**, es decir, sustituir su valor actual por el valor fijo Q3+3*RIC.

# df_pd.loc[df_pd["Sale_Price"] == Sale_Price, "Upper_Limit"] = ul
# df_pd.loc[df_pd["Sale_Price"] == Sale_Price, "Lower_Limit"] = ll

# df.loc[df["Sale_Price"]> (df["Sale_Price"].quantile(0.75) + (3 * (df["Sale_Price"].quantile(0.75)-df["Sale_Price"].quantile(0.25)))),"Sale_Price"]= (3 * (df["Sale_Price"].quantile(0.75)-df["Sale_Price"].quantile(0.25))) 

# COMMAND ----------

y = pd.to_numeric(df.Sale_Price, errors='coerce')

X=df.drop(['Sale_Price'],axis=1)

# COMMAND ----------

categorical_features_indices = np.where(X.dtypes != np.float)[0]

for pos in categorical_features_indices:
#     print(pos)
    colname = df.columns[pos]
    print(colname)
    df[[colname]] = df[[colname]].fillna('')
    df[[colname]] = df[[colname]].replace(np.nan,'',regex = True)


# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=4224, test_size=0.2)

# COMMAND ----------

# MAGIC %md # HYPERPARAMETER Tunning

# COMMAND ----------

# https://www.kaggle.com/code/egorphysics/catboost-hyperopt-on-meta-features/notebook

# COMMAND ----------

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# COMMAND ----------

with mlflow.start_run(run_name='CatBoostRegressor_Sale_Price') as run:

    if(run_hyperparameters == '1'):
        # search for best parameters
        best_params = fmin(
          fn=objective,
          space=search_space,
          algo=algorithm,
            max_evals=50) # 100
        #   max_evals=1000)

        # dict of the best params
        hyperparams = space_eval(search_space, best_params)

        params = {'learning_rate' : hyperparams['learning_rate'],
              'iterations' : hyperparams['iterations'],
              'depth' : hyperparams['depth'],
              'loss_function' : 'MAE',
              'l2_leaf_reg' : hyperparams['l2_leaf_reg'],
              'eval_metric' : 'MAE',
              'early_stopping_rounds': 100,
              'bootstrap_type' : hyperparams['bootstrap_type']}

    #     'loss_function' : 'RMSE',
    #     'eval_metric' : 'RMSE',

        print("Best params:")
        print(params)
        
        
    
    else: #run_hyperparameters == 0
        print("run_hyperparameters yes/no = ", run_hyperparameters)
        
        #Hard-coded - or read from files
        params = {'learning_rate' : 0.21,
              'iterations' : 950,
              'depth' : 9,
              'loss_function' : 'MAE',
              'l2_leaf_reg' : 5,
              'eval_metric' : 'MAE',
              'early_stopping_rounds': 100,
              'bootstrap_type' : 'Bernoulli'}
        

        
        
    model = CatBoostRegressor(**params, random_seed=42)

    model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test), verbose=False)
    
    pred_price = model.predict(X_test)    
    
    r2_score  =r2_score(y_test,pred_price)

    print("r2 score = ", r2_score)
    mlflow.log_metric('r2 score', r2_score)

    mae = mean_absolute_error(y_test,pred_price)
    
    print("mae = ", mae)
    
    mlflow.log_metric('mae', mae)
    
    df_pred = X_test 
    df_pred['SalePrice_Pred'] = pred_price
    df_pred['SalePrice_Pred'] = df_pred['SalePrice_Pred'].round(decimals = 0)
    df_pred['SalePrice_real'] = y_test
    df_pred['SalePrice_Diff'] = abs(df_pred['SalePrice_Pred'] - df_pred['SalePrice_real'])
    df_pred['SalePrice_Diff'] = df_pred['SalePrice_Diff'].round(decimals = 0)

# COMMAND ----------


# MAGIC %md ## Store the model

# COMMAND ----------

import pickle

with open(path_store_your_datalake + 'model_ML_Name_pickle.pkl', 'wb') as files:
    pickle.dump(model, files)

# COMMAND ----------

# MAGIC %md #Predictions

# COMMAND ----------

# preds = model.predict(test_set[cols])
pred_price = model.predict(X_test)

# COMMAND ----------

pred_price = model.predict(X_test)
print(pred_price)

# COMMAND ----------

from sklearn.metrics import r2_score

r2_score  =r2_score(y_test,pred_price)

print("r2 score = ", r2_score)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error
mae  =mean_absolute_error(y_test,pred_price)

print("mae = ", mae)

# COMMAND ----------

import seaborn as sns
sns.set(rc={'figure.figsize':(25,12)})
sns.regplot(x = y_test, y = pred_price)

# COMMAND ----------

df_pred = X_test 
df_pred['SalePrice_Pred'] = pred_price
df_pred['SalePrice_Pred'] = df_pred['SalePrice_Pred'].round(decimals = 0)
df_pred['SalePrice_real'] = y_test
df_pred['SalePrice_Diff'] = abs(df_pred['SalePrice_Pred'] - df_pred['SalePrice_real'])
df_pred['SalePrice_Diff'] = df_pred['SalePrice_Diff'].round(decimals = 0)

# COMMAND ----------

display(df_pred)

# COMMAND ----------

sns.stripplot(y='filed_X_visualization', x='SalePrice_Diff', data=df_pred) # , hue='anomaly'

# COMMAND ----------

sns.set(rc={'figure.figsize':(25,12)})

import matplotlib.pyplot as plt

sns.set_theme(style="ticks", palette="pastel")


# Draw a nested boxplot to show bills by day and time
ax = sns.boxplot(x="filed_X_visualization", y="SalePrice_Diff",
            data=df_pred)

# plt.ylim(0, 400)
ax.tick_params(axis='x', rotation=45) #rotae axis labels to read

plt.title('SalesPrice error by filed_X_visualization')

# COMMAND ----------


df_g = df_pred[["filed_X_visualization","SalePrice_Diff"]].groupby(['filed_X_visualization'], as_index=False).mean()

display(df_g)

# MAGIC %md ### Feature Importance

# COMMAND ----------

# https://www.rasgoml.com/feature-engineering-tutorials/how-to-generate-feature-importance-plots-using-catboost

# COMMAND ----------

feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Feature Importance')

# COMMAND ----------

# MAGIC %md ### Plot the permutation importance.

# COMMAND ----------

from sklearn.inspection import permutation_importance
# 

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=1066)
sorted_idx = perm_importance.importances_mean.argsort()
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Permutation Importance')


# COMMAND ----------

# MAGIC %md ## Plot the mean absolute value of the SHAP values

# COMMAND ----------

import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap_importance = shap_values.abs.mean(0).values
sorted_idx = shap_importance.argsort()
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), shap_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('SHAP Importance')


# COMMAND ----------


shap.plots.bar(shap_values, max_display=X_test.shape[0])

# COMMAND ----------



# COMMAND ----------

dbutils.notebook.exit("Success")