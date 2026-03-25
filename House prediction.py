import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split ,KFold,cross_validate,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.impute import SimpleImputer


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor

from sklearn.metrics import(
    mean_absolute_error ,
    root_mean_squared_error,
    r2_score
)   

# Configurations
pd.set_option('display.max_columns',None)
pd.set_option('display.float_format',lambda x : f"{x : 3f}")
sns.set_theme(style = 'darkgrid')


plt.rcParams.update({
    'axes.titlesize':10,
    'axes.labelsize': 9,
    'xtick.labelsize':8,
    'ytick.labelsize':8
})

RANDOM_STATE = 42
CSV_PATH = 'housing.csv'            # update path for a different dataset
TARGET_COL = 'median_house_value'   # target column name

# Load Data
df = pd.read_csv(CSV_PATH)
print('DataFrame shape:', df.shape)
print(df.head())

# Exploratory data Analysis
print(df.columns)
print(df.describe())
print(df.info())
print(df.isnull().sum())

num_cols = df.select_dtypes(include = [np.number]).columns.tolist()
cat_cols = df.select_dtypes(include =["object"]).columns.tolist()


print('target column', TARGET_COL)
print('Numerical columns',num_cols)
print('Categorical columns',cat_cols)

# missing value analysis

print('\Missing values per column:')
print(df.isna().sum())

# check presence of encoded missing values

for col in df.columns:
    print(df[col].value_counts().head(20))

# duplicates

duplicate_mask = df.duplicated()
num_duplicates = duplicate_mask.sum()
print('Number of duplicate rows:',num_duplicates) 

# (optional) drop duplicates if present

df = df.drop_duplicates()
print('shape after dropping duplicates:',df.shape)


# descriptive stat

print(df[num_cols].describe())

print(df[num_cols].describe().T)

# countplot for categorical columns

for col in cat_cols:
    plt.figure(figsize=(10,3))
    sns.countplot(x=col, data = df)
    plt.title(f'Distribution of {col}')
    plt.show()

# target column distribution
plt.figure(figsize = (6,4))
sns.histplot(df[TARGET_COL],bins=40,kde=True)
plt.title('Target Distribution: Median House value')
plt.xlabel('Median Hosue Value')
plt.show()


for col in df.columns:
    print(df[col].value_counts())

# # higher cap
print(df[TARGET_COL].value_counts())

# histogram plot
fig , axes = plt.subplots(3,3,figsize = (8,6))
axes = axes.flatten()

for i , col in  enumerate(num_cols):
    sns.histplot(df[col],kde = True , ax = axes[i])
    axes[i].set_title(col ,fontsize = 8)

plt.tight_layout()
plt.show()

# outliers analysis - boxplot
fig , axes = plt.subplots(3,3,figsize = (8,6))
axes = axes.flatten()

for i , col in enumerate(num_cols):
    sns.boxplot(x=df[col],ax=axes[i])
    axes[i].set_title(col,fontsize=8)
    axes[i].set_xlabel("")
plt.tight_layout()
plt.show()

# identify the presence of highly correlated columns & feature relationships 

plt.figure(figsize =(10,5))
sns.heatmap(
               df[num_cols].corr(),
               annot = True,
               cmap="coolwarm",
               center = 0
           )
plt.title('Correlation Heatmap')
plt.show()


# correlation with target 

corr_with_target = df[num_cols].corr()[TARGET_COL].sort_values(ascending=False)
print("\nCorrelation with Target:")
print(corr_with_target)


# Data Preprocessing

# seperate features and Target 
X = df.drop(columns = [TARGET_COL])
Y = df[TARGET_COL]

print(X.head())
print(Y.head())


# train test split 
X_train , X_test, Y_train ,Y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state= 42
)

print('Train shape:', X_train.shape)
print('Test shape:',X_test.shape)


# Preprocessing Pipeline 

numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print("NF:",numerical_features)
print("CF:",categorical_features)

# numerical features - preprocessing steps 

numerical_transformer = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
         ("scaler",StandardScaler())
    ]
    
)

# categorical features - preprocessing steps 


categorical_transformer = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
         ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ]
    
)


# preprocessing pipeline 
preprocess = ColumnTransformer(
transformers = [
    ('num', numerical_transformer,numerical_features),
    ('cat',categorical_transformer,categorical_features)
    ]

)

# Base line Model (No Cv Tuning)

baseline_pipe = Pipeline(
    steps =( 
        ('preprocess',preprocess),
        ('Model',LinearRegression())
    )
)

# preprocess the data & train baseline model 

baseline_pipe.fit(X_train,Y_train)

train_baseline_pred = baseline_pipe.predict(X_train)
test_baseline_pred = baseline_pipe.predict(X_test)

train_baseline_rmse = root_mean_squared_error(Y_train,train_baseline_pred)
train_baseline_mae = mean_absolute_error(Y_train,train_baseline_pred)
train_baseline_r2 = r2_score(Y_train,train_baseline_pred)

print("\n Train Baseline Metrics")
print(f'RMSE:{train_baseline_rmse:.3f}')
print(f'MAE:{train_baseline_mae:.3f}')
print(f'R2:{train_baseline_r2:.3f}')


test_baseline_rmse = root_mean_squared_error(Y_test,test_baseline_pred)
test_baseline_mae = mean_absolute_error(Y_test,test_baseline_pred)
test_baseline_r2 = r2_score(Y_test,test_baseline_pred)

print("\n Test Baseline Metrics")
print(f'RMSE:{test_baseline_rmse:.3f}')
print(f'MAE:{test_baseline_mae:.3f}')
print(f'R2:{test_baseline_r2:.3f}')


# Model selection & Optimization 
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=RANDOM_STATE),
    'Lasso': Lasso(random_state=RANDOM_STATE,max_iter=10000),
    'RandomForest': RandomForestRegressor(),
    'HistGB': HistGradientBoostingRegressor()
}


k = 5 
cv = KFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)

scoring ={
    'rmse':'neg_root_mean_squared_error', 
    'mae': 'neg_mean_absolute_error',
    'r2':'r2'
}

rows = []
for name , model in models.items():
    pipe = Pipeline(
        steps = (
            ('preprocess',preprocess),
            ('model',model)
        )
    )
    scores = cross_validate(pipe,X_train,Y_train,cv=cv,scoring=scoring,n_jobs=1)
    rows.append({
        'model': name,
        'cv_rmse': -scores['test_rmse'].mean(),
        'cv_mae':-scores['test_mae'].mean(),
        'cv_r2':-scores['test_r2'].mean()
    })

# sort based on lowest rmse value 
cv_results = pd.DataFrame(rows).sort_values('cv_rmse')
print(' CV model Comparison')
print(cv_results)
print(rows)


best_row = cv_results.sort_values('cv_rmse').iloc[0]


best_model_name = best_row['model']
best_rmse = best_row['cv_rmse']
print('CV RMSE:',best_rmse)


# Best Model = HistGradientBoostingRegressor


# Hyper parameter tunning 

hgb_pipe = Pipeline(
    steps =[
        ('preprocess',preprocess),
        ('model',HistGradientBoostingRegressor(random_state=RANDOM_STATE))
    ]
)

# Hyperparameter combinations 

param_grid = {
    'model__learning_rate': [0.003,0.05,0.1],
    'model__max_depth': [None,3,6],
    'model__max_leaf_nodes': [15,31,63],
    'model__min_samples_leaf': [20,50,100],
    'model__l2_regularization': [0.0,0.1,1.0]
}

grid = GridSearchCV(
    estimator=hgb_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# perform grid search 
grid.fit(X_train,Y_train)


print('\n TUNED HISTGB (CV)')
print('Best CV RMSE:',-grid.best_score_)
print('Best params:',grid.best_params_)


# Retraining with best params 


hgb_best = Pipeline(
    steps =[
        ('preprocess',preprocess),
        ('model',HistGradientBoostingRegressor(
            l2_regularization = 0.1,
            learning_rate = 0.1,
            max_depth = None,
            max_leaf_nodes = 63,
            min_samples_leaf=20
        ))

    ]
)

# train best model on entire training data (can also be done with refit = True in grid search) 

hgb_best.fit(X_train,Y_train)


# Final Evaluation 

train_final_pred = hgb_best.predict(X_train)

train_final_rmse = root_mean_squared_error(Y_train,train_final_pred)
train_final_mae = mean_absolute_error(Y_train,train_final_pred)
train_final_r2 = r2_score(Y_train,train_final_pred)

print("\n Final Model ( Tuned HGB) Train Performance")
print(f'RMSE:{train_final_rmse:.3f}')
print(f'MAE:{train_final_mae:.3f}')
print(f'R2:{train_final_r2:.3f}')



test_final_pred = hgb_best.predict(X_test)

test_final_rmse = root_mean_squared_error(Y_test,test_final_pred)
test_final_mae = mean_absolute_error(Y_test,test_final_pred)
test_final_r2 = r2_score(Y_test,test_final_pred)

print("\n Final Model ( Tuned HGB) Test Performance")
print(f'RMSE:{test_final_rmse:.3f}')
print(f'MAE:{test_final_mae:.3f}')
print(f'R2:{test_final_r2:.3f}')


# Residual Plot

residuals = Y_test - test_final_pred

plt.figure(figsize=(6,4))
plt.scatter(test_final_pred,residuals,s=10)
plt.axhline(0)
plt.title('Residuals vs Predictions')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()


plt.figure(figsize=(6,4))
sns.histplot(residuals,bins=42,kde=True)
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()


# Building a Predictive system

def predict_house_price(
        model,
        longitude: float,
        latitude: float,
        housing_median_age: float,
        total_rooms: float,
        total_bedrooms: float,
        population: float,
        households: float,
        median_income: float,
        ocean_proximity: str
    ) -> float:
    """
    Predict median_house_value for one new house.
    total_bedrooms can be np.nan (pipeline will impute).
    """
    new_row = pd.DataFrame([{
        'longitude': longitude, 
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,              
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }])

    return float(model.predict(new_row)[0])


# Example Inference
example_pred = predict_house_price(
    model=hgb_best,
    longitude=-122.230,
    latitude=37.880,
    housing_median_age=41,
    total_rooms=880,
    total_bedrooms=129,
    population=322,
    households=126,
    median_income=8.3252,
    ocean_proximity="NEAR BAY"
)

print('\nExample prediction:', round(example_pred, 2))


