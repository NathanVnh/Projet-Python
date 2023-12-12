##Installation des packages nécessaires pour l'analyse
from pynsee.macrodata.get_series import get_series
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.stattools import coint
##Connexion avec l'API INSEE
from pynsee import init_conn


class ProjetPython:
    def __init__(self):
        self.f_time="TIME_PERIOD"
        self.f_value="OBS_VALUE" 
        self.f_value_x="OBS_VALUE_x" 
        self.f_value_y="OBS_VALUE_y" 
        init_conn(insee_key='OhPJjhlU6BcU1jgxYWzIWq1RcUka', insee_secret='pjqgGXAy7co3cMD4zZ8aytN5tq4a')
        
    def import1(self,idbank):
        df_init = get_series(idbank)
        df = df_init.loc[:,['TIME_PERIOD','OBS_VALUE']]
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
        df.set_index('TIME_PERIOD',inplace=True)
        return df
        
    def import2(self,idbank,name):
        df_init = get_series(idbank)
        df = df_init.loc[:,['TIME_PERIOD','OBS_VALUE']]
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
        df.set_index('TIME_PERIOD',inplace=True)
        df["Series"]=name
        return df

    def sub_rename(self,df,time,name):
        df = df[df.index>time]
        df = df.rename(columns={'OBS_VALUE': name})
        return df


