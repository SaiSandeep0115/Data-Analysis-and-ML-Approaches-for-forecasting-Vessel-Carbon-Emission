#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


import pandas as pd
import numpy as np

df_train = pd.read_csv("data/train.csv")


# In[2]:


df_train.head()


# In[3]:


df_train.info()


# In[4]:


df_test = pd.read_csv("data/test.csv")
df_test.head()


# In[5]:


df_test.info()


# # Raw Data Exploration - Train and Test 

# ### Information

# In[6]:


raw_train = df_train.copy()
raw_train.head()


# In[7]:


raw_test = df_test.copy()
raw_test.head()


# In[8]:


nuniques = raw_train.nunique()
print(nuniques)


# In[9]:


nuniques = raw_test.nunique()
print(nuniques)


# In[10]:


raw_train.describe()


# In[11]:


raw_test.describe()


# ### Pre-processing for comparing train and test data pattern

# In[12]:


from sklearn.preprocessing import LabelEncoder

raw_train = raw_train.drop(columns=['IMO', 'NAME', 'REGISTERED'])

raw_train['EFFICIENCY_TYPE'] = raw_train['EFFICIENCY'].apply(lambda x: 1 if 'EIV' in x else 2 if 'EEDI' in x else 3)

raw_train['EFFICIENCY_VALUE'] = raw_train['EFFICIENCY'].str.extract(r'([\d.]+)').astype(float)

raw_train['EFFICIENCY_VALUE'].fillna(raw_train['EFFICIENCY_VALUE'].mean(), inplace=True)

raw_train['TYPE'] = LabelEncoder().fit_transform(raw_train['TYPE'])

raw_train = raw_train.drop(columns=['EFFICIENCY'])

raw_train.head()


# In[13]:


raw_test = raw_test.drop(columns=['IMO', 'NAME', 'REGISTERED'])

raw_test['EFFICIENCY_TYPE'] = raw_test['EFFICIENCY'].apply(lambda x: 1 if 'EIV' in x else 2 if 'EEDI' in x else 3)

raw_test['EFFICIENCY_VALUE'] = raw_test['EFFICIENCY'].str.extract(r'([\d.]+)').astype(float)

raw_test['EFFICIENCY_VALUE'].fillna(raw_test['EFFICIENCY_VALUE'].mean(), inplace=True)

raw_test['TYPE'] = LabelEncoder().fit_transform(raw_test['TYPE'])

raw_test = raw_test.drop(columns=['EFFICIENCY'])

raw_test.head()


# ### Correlation

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(raw_train.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(raw_test.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()


# ### Regression Analysis

# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))

columns_to_plot = ['TYPE', 'BUILD_YEAR', 'GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH', 'EFFICIENCY_TYPE', 'EFFICIENCY_VALUE']

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.regplot(x=raw_train[col], y=raw_train['EMISSION'], line_kws={"color": "red"})
    plt.title(f'Regression: EMISSION vs {col}')
    plt.xlabel(col)
    plt.ylabel('EMISSION')

plt.tight_layout()
plt.show()


# ### Outlier Detection

# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

plt.figure(figsize=(15, 15))

columns_to_plot = ['TYPE', 'BUILD_YEAR', 'GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH', 'EFFICIENCY_TYPE', 'EFFICIENCY_VALUE', 'EMISSION']

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    
    Q1 = raw_train[col].quantile(0.25)
    Q3 = raw_train[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = raw_train[(raw_train[col] < lower_bound) | (raw_train[col] > upper_bound)]
    
    sns.boxplot(x=raw_train[col], color='lightblue')
    
    if len(outliers) > 0:
        sns.scatterplot(x=outliers[col], y=[0]*len(outliers), color='red', marker='x', label=f'Outliers ({len(outliers)})')
        plt.legend(title='Outliers', loc='upper left')
    
    plt.title(f'Outlier Detection: {col}')
    plt.xlabel(col)
    
plt.tight_layout()
plt.show()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

plt.figure(figsize=(15, 15))

columns_to_plot = ['TYPE', 'BUILD_YEAR', 'GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH', 'EFFICIENCY_TYPE', 'EFFICIENCY_VALUE']

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    
    Q1 = raw_test[col].quantile(0.25)
    Q3 = raw_test[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = raw_test[(raw_test[col] < lower_bound) | (raw_test[col] > upper_bound)]
    
    sns.boxplot(x=raw_test[col], color='lightblue')
    
    if len(outliers) > 0:
        sns.scatterplot(x=outliers[col], y=[0]*len(outliers), color='red', marker='x', label=f'Outliers ({len(outliers)})')
        plt.legend(title='Outliers', loc='upper left')
    
    plt.title(f'Outlier Detection: {col}')
    plt.xlabel(col)
    
plt.tight_layout()
plt.show()


# ### Distribution Analysis

# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

plt.figure(figsize=(15, 15))

columns_to_plot = ['TYPE', 'BUILD_YEAR', 'GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH', 'EFFICIENCY_TYPE', 'EFFICIENCY_VALUE', 'EMISSION']

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    
    sns.histplot(raw_train[col], kde=True, color='lightblue', bins=20)
    
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
plt.tight_layout()
plt.show()


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

plt.figure(figsize=(15, 15))

columns_to_plot = ['TYPE', 'BUILD_YEAR', 'GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH', 'EFFICIENCY_TYPE', 'EFFICIENCY_VALUE']

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    
    sns.histplot(raw_test[col], kde=True, color='lightblue', bins=20)
    
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
plt.tight_layout()
plt.show()


# ### Removing 10 Extreme Outliers for Raw_Data for understanding patterns better

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

columns_to_plot = ['TYPE', 'BUILD_YEAR', 'GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH', 'EFFICIENCY_TYPE', 'EFFICIENCY_VALUE', 'EMISSION']

plt.figure(figsize=(20, 20))

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    
    Q1 = raw_train[col].quantile(0.25)
    Q3 = raw_train[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = raw_train[(raw_train[col] < lower_bound) | (raw_train[col] > upper_bound)]
    
    outliers['distance_from_median'] = outliers[col].apply(lambda x: abs(x - raw_train[col].median()))
    
    extreme_outliers = outliers.nlargest(10, 'distance_from_median')
    
    raw_train = raw_train[~raw_train.index.isin(extreme_outliers.index)]
    
    sns.boxplot(x=raw_train[col], color='lightblue')
    
    plt.title(f'Outlier Detection (Excluding 10 Extreme Outliers): {col}')
    plt.xlabel(col)
    
plt.tight_layout()
plt.show()


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))

columns_to_plot = ['TYPE', 'BUILD_YEAR', 'GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH', 'EFFICIENCY_TYPE', 'EFFICIENCY_VALUE']

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.regplot(x=raw_train[col], y=raw_train['EMISSION'], line_kws={"color": "red"})
    plt.title(f'Regression: EMISSION vs {col}')
    plt.xlabel(col)
    plt.ylabel('EMISSION')

plt.tight_layout()
plt.show()


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

plt.figure(figsize=(15, 15))

columns_to_plot = ['TYPE', 'BUILD_YEAR', 'GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH', 'EFFICIENCY_TYPE', 'EFFICIENCY_VALUE', 'EMISSION']

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    
    sns.histplot(raw_train[col], kde=True, color='lightblue', bins=20)
    
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
plt.tight_layout()
plt.show()


# In[24]:


plt.figure(figsize=(10,7))
correlation_matrix = raw_train.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# ### Multi-collinearity 

# In[25]:


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_train = raw_train.select_dtypes(include=[np.number])
X_train = add_constant(X_train)

vif_train = pd.DataFrame()
vif_train["Variable"] = X_train.columns
vif_train["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

print("VIF for raw_train:")
print(vif_train)

X_test = raw_test.select_dtypes(include=[np.number])
X_test = add_constant(X_test)

vif_test = pd.DataFrame()
vif_test["Variable"] = X_test.columns
vif_test["VIF"] = [variance_inflation_factor(X_test.values, i) for i in range(X_test.shape[1])]

print("\nVIF for raw_test:")
print(vif_test)


# ### Scatterplot of Emissions vs Efficiency values WRTO Efficiency Types

# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns

efficiency_type_mapping = {1: 'EIV', 2: 'EEDI', 3: 'Eff-NA'}
raw_train['EFFICIENCY_TYPE_LABEL'] = raw_train['EFFICIENCY_TYPE'].map(efficiency_type_mapping)

plt.figure(figsize=(12, 8))

sns.scatterplot(data=raw_train, 
                x='EFFICIENCY_VALUE', 
                y='EMISSION', 
                hue='EFFICIENCY_TYPE_LABEL', 
                palette='Set1',  
                s=20, 
                edgecolor='w',   
                marker='o')

plt.title('EMISSION vs EFFICIENCY_VALUE', fontsize=16)
plt.xlabel('Efficiency Value', fontsize=14)
plt.ylabel('Emission', fontsize=14)

plt.legend(title='EFFICIENCY_TYPE', loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()


# ### Raw Data Exploration - Insights
# 
# **Regression Analysis:** Not a good fit but a few features has linearity with Emission
# 
# **Outlier Detection:** Many outliers for almost all features but few are significant 
# 
# **Correlation Analysis:** No highly correlated features but 3 features have notable correlation with Emission
# 
# **Distribution B/W Train and Test:** Train and Test data has similar distributions and patterns
# 
# **Multi-Collinearity:** Moderate Multi-collinearity is present
# 
# **Scatter of Emissions vs Eff Values WRTO Eff Type**: No Significant Pattern change WRTO Efficiency Type
# 
# **No Missing Values**
# 
# 
# ### Raw Data Exploration - Key Takeaways
# 
# **Removing only extreme outliers for train data to remain the data distribution similar to test data**
# 
# **Considering ML Models which can handle the remaining outliers for better prediction**
# 
# **Minding the moderate multicollinearity during modeling**

# # Data Pre-Processing

# In[27]:


data = df_train.copy()
test = df_test.copy()


# In[28]:


data.head()


# In[29]:


data.info()


# In[30]:


data.describe()


# ### Handling Efficiency Column

# In[31]:


eiv_count = data['EFFICIENCY'].str.contains('EIV', na=False).sum()
eedi_count = data['EFFICIENCY'].str.contains('EEDI', na=False).sum()
na_count = data['EFFICIENCY'].str.contains('Not Applicable', na=False).sum()

total_count = eiv_count + eedi_count + na_count

print(f"EIV count: {eiv_count}")
print(f"EEDI count: {eedi_count}")
print(f"Not Applicable count: {na_count}")
print(f"Total count: {total_count}")


# In[32]:


# Processing Efficiency Columns
def process_efficiency(df):
    # One-hot encoded columns
    df['EIV'] = df['EFFICIENCY'].str.contains('EIV', na=False).astype(int)
    df['EEDI'] = df['EFFICIENCY'].str.contains('EEDI', na=False).astype(int)
    df['Eff-NA'] = df['EFFICIENCY'].str.contains('Not Applicable', na=False).astype(int)

    df['EFFICIENCY_VALUES'] = df['EFFICIENCY'].str.extract(r'([\d\.]+)').astype(float)

    mean_efficiency = df['EFFICIENCY_VALUES'].mean(skipna=True)
    df['EFFICIENCY_VALUES'].fillna(mean_efficiency, inplace=True)
    
    return df

data = process_efficiency(data)
test = process_efficiency(test)


# ### Encoding and Removal of Unnecessary Columns

# In[33]:


def process_data(df):
    df.drop(['IMO', 'NAME', 'EFFICIENCY', 'REGISTERED'], axis=1, inplace=True)
    
    df = pd.get_dummies(df, columns=['TYPE'], drop_first=True)
    
    boolean_cols = df.select_dtypes(include=[bool]).columns
    df[boolean_cols] = df[boolean_cols].astype(int)
    
    return df

data = process_data(data)
test = process_data(test)
data.shape


# ### Outlier Removal (Only Extreme 100 Data Points)

# In[34]:


import pandas as pd
import numpy as np
from scipy.stats import zscore

outlier_cols = ['GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH', 'EMISSION', 'EFFICIENCY_VALUES']

def remove_top_outliers(df, cols, n):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_clean[(df_clean[col] > upper_bound) | (df_clean[col] < lower_bound)]
        outliers = outliers.sort_values(by=col, ascending=False).head(n)
        df_clean = df_clean.drop(outliers.index)

    return df_clean

data = remove_top_outliers(data, outlier_cols, n=100)
data.shape


# ### PCA, Feature Engineering and Selection - Not Significant

# In[35]:


data.head()


# In[36]:


data.columns


# In[37]:


data.shape


# # Modeling

# ### Linear Regression, Lasso, Ridge, SVR, KNN, 
# ### Decision Tree, Random Forest, AdaB, CatB

# In[38]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore")

X = data.drop(columns=['EMISSION'], axis=1)
y = data['EMISSION']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'RMSE': rmse, 'R² Score': r2}
    print(f"{model_name} -> RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

# Linear Regression
evaluate_model(LinearRegression(), X_train, X_test, y_train, y_test, "Linear Regression")

# Lasso Regression
evaluate_model(Lasso(alpha=0.1), X_train_scaled, X_test_scaled, y_train, y_test, "Lasso Regression")

# Ridge Regression
evaluate_model(Ridge(alpha=1.0), X_train_scaled, X_test_scaled, y_train, y_test, "Ridge Regression")

# Support Vector Regression
evaluate_model(SVR(kernel='rbf'), X_train_scaled, X_test_scaled, y_train, y_test, "SVM Regression")

# K-Nearest Neighbors Regression
evaluate_model(KNeighborsRegressor(n_neighbors=5), X_train_scaled, X_test_scaled, y_train, y_test, "KNN Regression")

# Decision Tree Regression
evaluate_model(DecisionTreeRegressor(max_depth=5), X_train, X_test, y_train, y_test, "Decision Tree Regression")

# Random Forest Regression
evaluate_model(RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test, "Random Forest Regression")

# AdaBoost Regression
evaluate_model(AdaBoostRegressor(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test, "AdaBoost Regression")

# CatBoost Regression
evaluate_model(CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=0), 
               X_train, X_test, y_train, y_test, "CatBoost Regression")


# ### Gradient and Extreme Gradient Boosting

# In[39]:


from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

best_params_gb = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
best_params_xgb = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}

best_gb_model = GradientBoostingRegressor(**best_params_gb, random_state=42)
best_gb_model.fit(X_train, y_train)
y_pred_best_gb = best_gb_model.predict(X_test)

best_xgb_model = XGBRegressor(**best_params_xgb, random_state=42)
best_xgb_model.fit(X_train, y_train)
y_pred_best_xgb = best_xgb_model.predict(X_test)

rmse_best_gb = np.sqrt(mean_squared_error(y_test, y_pred_best_gb))
r2_best_gb = r2_score(y_test, y_pred_best_gb)
rmse_best_xgb = np.sqrt(mean_squared_error(y_test, y_pred_best_xgb))
r2_best_xgb = r2_score(y_test, y_pred_best_xgb)

print(f"Best Gradient Boosting RMSE: {rmse_best_gb:.4f}")
print(f"Best Gradient Boosting R² Score: {r2_best_gb:.4f}")
print(f"Best XGBoost RMSE: {rmse_best_xgb:.4f}")
print(f"Best XGBoost R² Score: {r2_best_xgb:.4f}")


# ### Random Forest Hypertuned

# In[40]:


from sklearn.ensemble import RandomForestRegressor

best_rf_model = RandomForestRegressor(
    max_depth=18, 
    min_samples_leaf=1, 
    min_samples_split=10, 
    n_estimators=180, 
    random_state=42
)

best_rf_model.fit(X_train, y_train)

y_pred_best_rf = best_rf_model.predict(X_test)

mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
rmse_best_rf = np.sqrt(mse_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

print(f"Best Random Forest RMSE: {rmse_best_rf:.4f}")
print(f"Best Random Forest R² Score: {r2_best_rf:.4f}")


# ### Cat Boost Hypertuned

# In[41]:


from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

best_catboost = CatBoostRegressor(
    depth=8,
    iterations=800,
    l2_leaf_reg=3,
    learning_rate=0.05,
    loss_function='RMSE',
    random_state=42,
    verbose=0
)

best_catboost.fit(X_train, y_train)

y_pred_best_catboost = best_catboost.predict(X_test)

rmse_best_catboost = np.sqrt(mean_squared_error(y_test, y_pred_best_catboost))
r2_best_catboost = r2_score(y_test, y_pred_best_catboost)

print(f"Best CatBoost RMSE: {rmse_best_catboost:.4f}")
print(f"Best CatBoost R² Score: {r2_best_catboost:.4f}")


# ### Catboost K-Fold

# In[42]:


from sklearn.model_selection import cross_val_score, KFold
from catboost import CatBoostRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np

best_catboost = CatBoostRegressor(
    depth=8,
    iterations=800,
    l2_leaf_reg=3,
    learning_rate=0.05,
    loss_function='RMSE',
    random_state=42,
    verbose=0
)

cv = KFold(n_splits=10, shuffle=True, random_state=42)

rmse_scorer = make_scorer(mean_squared_error, squared=False)  # RMSE
r2_scorer = make_scorer(r2_score)  

rmse_scores = cross_val_score(best_catboost, X, y, cv=cv, scoring=rmse_scorer)
r2_scores = cross_val_score(best_catboost, X, y, cv=cv, scoring=r2_scorer)

print(f"Cross-Validation RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Cross-Validation R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")


# ### Deep Neural Network

# In[43]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def build_advanced_dnn():
    model = Sequential([
        Dense(256, input_shape=(X_train_scaled.shape[1],)), 
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.4),

        Dense(128),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.3),

        Dense(64),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.3),

        Dense(32),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1)
    ])


    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mse'])
    
    return model


dnn_model_advanced = build_advanced_dnn()


early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)


history = dnn_model_advanced.fit(X_train_scaled, y_train, 
                                 validation_data=(X_test_scaled, y_test),
                                 epochs=200, batch_size=64, 
                                 callbacks=[early_stopping, reduce_lr], 
                                 verbose=1)


y_pred_dnn_advanced = dnn_model_advanced.predict(X_test_scaled).flatten()


mse_dnn_adv = mean_squared_error(y_test, y_pred_dnn_advanced)
rmse_dnn_adv = np.sqrt(mse_dnn_adv)
r2_dnn_adv = r2_score(y_test, y_pred_dnn_advanced)

print(f"Optimized DNN Model RMSE: {rmse_dnn_adv:.4f}")
print(f"Optimized DNN Model R² Score: {r2_dnn_adv:.4f}")


# ### Tabnet Architecture

# In[44]:


from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler
import torch

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

tabnet_model = TabNetRegressor()
tabnet_model.fit(
    X_train=X_train_scaled, y_train=y_train.values.reshape(-1, 1),
    eval_set=[(X_test_scaled, y_test.values.reshape(-1, 1))],
    patience=20, max_epochs=200
)

y_pred_tabnet = tabnet_model.predict(X_test_scaled).flatten()

mse_tabnet = mean_squared_error(y_test, y_pred_tabnet)
rmse_tabnet = np.sqrt(mse_tabnet)
r2_tabnet = r2_score(y_test, y_pred_tabnet)

print(f"TabNet RMSE: {rmse_tabnet:.4f}")
print(f"TabNet R² Score: {r2_tabnet:.4f}")


# # Prediction

# In[45]:


import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

best_catboost = CatBoostRegressor(
    depth=8,
    iterations=800,
    l2_leaf_reg=3,
    learning_rate=0.05,
    loss_function='RMSE',
    random_state=42,
    verbose=0
)

best_catboost.fit(X_train, y_train)

y_pred_best_catboost = best_catboost.predict(X_test)

rmse_best_catboost = np.sqrt(mean_squared_error(y_test, y_pred_best_catboost))
r2_best_catboost = r2_score(y_test, y_pred_best_catboost)

print(f"Best CatBoost RMSE: {rmse_best_catboost:.4f}")
print(f"Best CatBoost R² Score: {r2_best_catboost:.4f}")

y_pred_test = best_catboost.predict(test)

output_df = pd.DataFrame({
    'ID': test.index, 
    'Predicted': y_pred_test
})

output_df.to_csv("predicted_emissions.csv", index=False)

print("Predictions saved to predicted_emissions.csv")

