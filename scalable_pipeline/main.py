import pandas as pd

columns = ["sepal_length","sepal_width","petal_length","petal_width","class"]

df = pd.read_csv("iris.data",header=None)

df.columns = columns

print(df)

classes = df['class'].unique()



for c in classes:
    df_sub = df[df['class']==c]