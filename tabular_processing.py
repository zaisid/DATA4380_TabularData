
import pandas as pd

def remove_name_col(df, col="name"):
    df = df.drop(col,axis=1)
    return df

def balance_classes(df):
    oversample = pd.concat([ df[df["status"]==0], df[df["status"]==0] ], axis=0, ignore_index=True)
    df1 = pd.concat([df,oversample], axis=0, ignore_index=True)
    return df1

def minmax_scaling(df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns = df.columns)
    return scaled_df

def preprocess(df):
    df = remove_name_col(df)
    df = balance_classes(df)
    df = minmax_scaling(df)
    return df
