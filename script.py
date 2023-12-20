class ProjetPython:
    '''
    Classe qui permet de gérer l'import des données via l'API BDM INSEE en connaissant les idbank, et qui permet certaines mises au format Séries temporelles pour rendre le code moins chargé.
    '''
    def __init__(self):
        '''
        Parameters
        ----------
        self.f_time : str
        contient le nom de la variable dans les tables INSEE qui indique la période
        self.f_value : str
        contient le nom de la variable dans les tables INSEE qui indique la valeur numérique de l'indice récupéré
        self.f_value_x : str
        pour faciliter les fusions des tables Insee qui contiennent le même nom de variable
        self.f_value_y : str
        pour faciliter les fusions des tables Insee qui contiennent le même nom de variable
        '''     
        
        self.f_time="TIME_PERIOD"
        self.f_value="OBS_VALUE" 
        self.f_value_x="OBS_VALUE_x" 
        self.f_value_y="OBS_VALUE_y" 
        
    def import1(self,idbank):
        '''
        Méthode qui permet d'importer des données directement depuis l'API Insee en connaissant l'Idbank de la série avec quelques étapes de mise en forme (sélection des colonnes, et mise au format Séries Temporelles)

        
        Parameters
        ----------
        idbank : str
        l'IDBANK correspondant à la série que l'on souhaite importer (à identifier directement sur le site de l'INSEE
        '''    
        df_init = get_series(idbank)
        df = df_init.loc[:,['TIME_PERIOD','OBS_VALUE']]
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
        df.set_index('TIME_PERIOD',inplace=True)
        return df
        
    def import2(self,idbank,name):
        '''
        Méthode qui permet d'importer des données directement depuis l'API Insee en connaissant l'Idbank de la série avec quelques étapes de mise en forme (sélection des colonnes, et mise au format Séries Temporelles)

        
        Parameters
        ----------
        idbank : str
        l'IDBANK correspondant à la série que l'on souhaite importer (à identifier directement sur le site de l'INSEE
        name : str
        Le nom de la série considéré (pour les affichages graphiques)
        '''    
        df_init = get_series(idbank)
        df = df_init.loc[:,['TIME_PERIOD','OBS_VALUE']]
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
        df.set_index('TIME_PERIOD',inplace=True)
        df["Series"]=name
        return df

    def sub_rename(self,df,time,name):
        '''
        Méthode de mise en forme de data_frame, qui réduit la plage temporelle considéré et renomme la colonne contenant les valeurs numériques de l'indice avec le nom souhaité
        
        Parameters
        ----------
        df : DataFrame
        Le DataFrame que l'on souhiate modifier
        time : str
        La date à partir de laquelle on travaille
        name : str
        Le nom souhaité pour la colonne contenant les valeurs (pour les affichages graphiques)
        '''
        df = df[df.index>time]
        df = df.rename(columns={'OBS_VALUE': name})
        return df
