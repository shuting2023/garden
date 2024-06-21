from shapely.geometry import Point
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import pandas as pd
import numpy as np
import re

state_df = gpd.read_file('data/raw/cb_2018_us_state_20m/cb_2018_us_state_20m.shp')

def clean_mh_data(file_path, drop_col =['Population2010', 'MHLTH_Adj95CI','Geolocation']):
    df = pd.read_csv(file_path)
    df = df.drop(columns = drop_col)
    return df

def clean_gs_data(file_path, keep_col):
    df = pd.read_csv(file_path, encoding="unicode_escape",low_memory=False)
    df.dropna(how = 'any', inplace = True)
    df = df[df['CTR_MN_NM'] == 'United States']
    df['ID_HDC_G0'] = df['ID_HDC_G0'].astype(int)
    return df[keep_col]

def check_string_format(df, re_pattern = r"^[a-zA-Z0-9\s.,;'\-\(\)&/]+$"):
    """
    Identify and display probelmatic columns in a dataframe based on a regular expression pattern.
    """
    acceptable_pattern = re.compile(re_pattern)
    for col in df.columns:
        unacceptable = df[~df[col].astype(str).str.contains(acceptable_pattern)]
        if len(unacceptable) > 0:
            print(col, len(unacceptable))
            display(unacceptable)
    return None

def format_clean(df, col_list = ['UC_NM_MN', 'UC_NM_LST']):
    """
    Clean and format columns in a dataframe based on a regular expression
    """

    df[col_list] = df[col_list].apply(lambda x: x.str.replace(r'?', "'"))
    df[col_list] = df[col_list].apply(lambda x: x.str.replace(r'[\[\]]', "", regex = True))
    df[col_list] = df[col_list].apply(lambda x: x.str.replace('ÿ', " "))
    return df 

def label_state(df):
    df['State'] = df.apply(state_identifier, axis = 1)
    return df

def state_identifier(row, state_df = state_df):
    centroid = Point(row['GCPNT_LON'], row['GCPNT_LAT'])
    state = state_df[state_df.contains(centroid)]['STUSPS'].values[0]
    if state:
        return state
    else:
        return np.nan

def avg_per_person(df, cols):
    df[cols] = df[cols].astype(float)  
    new_cols = []
    for col in cols:
        df[col + '_AV'] = df[col] / df['P15']
        new_cols.append(col + '_AV')
    df.drop(columns = cols, inplace = True)
    return df, new_cols

def convert_to_float(df, cols):
    for col in cols:
        df[col] = df[col].astype(float)
    return df

def match_and_merge(row,df2):
    """
    Helping function that returns value from a dataframe based on a match between two columns.
    """
    match = df2[(df2['StateAbbr'] == row['State']) & (df2['PlaceName'].apply(lambda x: x in row['PlaceName']))]
    if len(match) > 0:
        return match['MHLTH_AdjPrev'].values[0]
    else:
        return np.nan

def merge_mh_gs(df,df2):
    """
    Merges two dataframes based on a match between the 'State' and 'PlaceName' columns.
    """
    df['UC_NM_LST'] = df['UC_NM_LST'].astype(str)
    df['PlaceName'] = df['UC_NM_LST'].apply(lambda x: x.split(';'))
    df = df.explode('PlaceName')
    df['AVG_NGMH_ADJPREV'] = df.apply(lambda x: match_and_merge(row=x, df2=df2), axis = 1)
    df.dropna(subset = ['AVG_NGMH_ADJPREV'], inplace = True)
    df.drop(columns = ['PlaceName', 'GCPNT_LAT','GCPNT_LON'], inplace = True)
    return df.groupby(df.columns.tolist()).agg('mean').reset_index()

# change the name to heatmap later
def heatmap(df,
    figsize=(18, 2),
    annot_fontsize = 6,
    rotation = 45,
    xticks_fontsize = 8,
    title = 'Spearman Correlation between Numberic Urban Features and AVG_NGMH_ADJPREV'):
    _ = plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f", cbar = False ,linewidths=0.5, annot_kws={"size":annot_fontsize})
    plt.title(label = title, fontdict={'fontsize': 10})
    plt.xlabel('Urban Features', fontdict={'fontsize': 10})
    plt.yticks(fontsize = 8)
    plt.xticks(fontsize =xticks_fontsize, rotation = rotation)
    plt.show()
    return None

def remove_outliers(df, col):
    df[col] = df[col].astype(float)
    mu = df[col].mean()
    sd = df[col].std()
    return df[(df[col]> mu - 3 * sd) & (df[col] < mu + 3 * sd)]

def assemble_corr_df(df, col_list):
    corr_dict = {}
    indf = remove_outliers(df,'AVG_NGMH_ADJPREV')
    for col in col_list:
        remove_df = remove_outliers(indf, col)
        corr = remove_df[col].corr(remove_df['AVG_NGMH_ADJPREV'], method='spearman')
        if np.isnan(corr):
            corr = 0
        corr_dict[col] = corr
    corr_df = pd.DataFrame(corr_dict, index = ['AVG_NGMH_ADJPREV']).drop(['AVG_NGMH_ADJPREV'], axis = 1).T
    return corr_df.sort_values(by = 'AVG_NGMH_ADJPREV', ascending = False)

def r2_score_df(df, cols, y_col):
    r2_score = {}
    for col in cols:
        model = LinearRegression()
        indf = remove_outliers(df, col)
        x = indf[col].values.reshape(len(indf[col]), 1)
        y = indf[y_col].values.reshape(len(indf[col]), 1)
        model.fit(x,y)
        r2_score[col] = model.score(x,y)
    r2_df = pd.DataFrame(r2_score, index = ['R2']).T
    return r2_df

def heatmap_r2(df, figsize=(18, 1), title = 'Urban Features vs Mental Health', label_fontsize = 10, ticks_fontsize = 8):
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f", cbar = False ,linewidths=0.5, annot_kws={"size":6}, vmin = 0, vmax = 1, center = 0.5)
    plt.title(label=title, fontdict={'fontsize': label_fontsize})
    plt.xticks(fontsize = ticks_fontsize)
    plt.yticks(fontsize = ticks_fontsize)
    plt.show()
    return None

def extract_normalize(df, cols):
    num_df = df[cols]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(num_df)
    scaled_df = pd.DataFrame(scaled_data, columns=cols)
    return scaled_df

def multi_linear_reg(df, x_cols, y_col):
    X = df[x_cols]
    y = df[y_col]
    lm = LinearRegression()
    model = lm.fit(X, y)

    coef_dic = {}
    for idx, col in enumerate(x_cols):
        coef_dic[col] = model.coef_[idx]
    
    coef_df = pd.DataFrame(coef_dic, index = ['coef']).T
    coef_df = coef_df.sort_values(by = 'coef', ascending = False)
    return coef_df, model.score(X, y)

def urban_cols(original_cols, remove_col):
    urban_cols = original_cols.copy()
    urban_cols.remove(remove_col)
    return urban_cols

def one_func_multi_reg_urban(original_cols, remove_col, df, title):
    cols = urban_cols(original_cols, remove_col)
    coef_df,_= multi_linear_reg(df = df, x_cols=cols, y_col = remove_col)
    heatmap(coef_df.T, figsize=(18, 1), title = title)
    return None

def mediation_analysis(df, x_col, mediator_col, y_col):    
    # mediator model: Regress mediator on independent variable
    mediator_model = sm.OLS.from_formula(f"{mediator_col} ~ {x_col}", data=df)

    # outcome model: Regress outcome on independent variable and mediator
    outcome_model = sm.OLS.from_formula(f"{y_col} ~ {x_col} + {mediator_col}", data=df)

    # mediation model: Regress outcome on mediator
    mediation_model = Mediation(outcome_model, mediator_model, x_col, mediator_col)
    med_result = mediation_model.fit()

    return med_result.summary()