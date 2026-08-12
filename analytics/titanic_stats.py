import seaborn as sns
import pandas as pd

df = sns.load_dataset('titanic')
print((df.isnull().mean() * 100).sort_values(ascending=False))
print() 
print(df[['age','embarked','deck','fare']].dtypes)
