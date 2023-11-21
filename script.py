##Packages possibles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from datetime import datetime

##Récupération des données
pip install pynsee
from pynsee.macrodata.get_series_list import get_series_list
from pynsee.macrodata.get_series import get_series
from pynsee import init_conn
init_conn(insee_key='OhPJjhlU6BcU1jgxYWzIWq1RcUka', insee_secret='pjqgGXAy7co3cMD4zZ8aytN5tq4a')
#On récupère les prix à l'importation de l'énergie via l'API
energie_prix_import = get_series('010535859')
##Idem pour l'Indice des Prix de Production total Industrie et l'Indice des Prix à la Consommation total
IPP_tot = get_series('010535587')
IPC_tot = get_series('001759970')

##Premières analyses des séries temporelles
energie_prix_import = energie_prix_import.loc[:,['TIME_PERIOD','OBS_VALUE']]
energie_prix_import['TIME_PERIOD'] = pd.to_datetime(energie_prix_import['TIME_PERIOD'])
energie_prix_import.set_index('TIME_PERIOD',inplace=True)
result = seasonal_decompose(energie_prix_import,model='additive')
trend = result.trend
seasonal = result.seasonal
residual = result.resid
##Affichage graphique
plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(energie_prix_import, label="initial")
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label="trend")
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label="seasonal")
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label="residual")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
##Forcément on a un modèle de séries temporelles qui capte mal la crise énergétique sur la péridoe de la guerre en Ukraine

##On transforme également IPP et IPC en séries temporelles IPP
IPP_tot = IPP_tot_insee.loc[:,['TIME_PERIOD','OBS_VALUE']]
IPP_tot['TIME_PERIOD'] = pd.to_datetime(IPP_tot['TIME_PERIOD'])
IPP_tot.set_index('TIME_PERIOD',inplace=True)
IPC_tot = IPC_tot_insee.loc[:,['TIME_PERIOD','OBS_VALUE']]
IPC_tot['TIME_PERIOD'] = pd.to_datetime(IPC_tot['TIME_PERIOD'])
IPC_tot.set_index('TIME_PERIOD',inplace=True)
##Test de stationnarité des séries temporelles
energy_adf = adfuller(energie_prix_import)
IPP_tot_adf = adfuller(IPP_tot)
IPC_tot_adf = adfuller(IPC_tot)
print('p-value',energy_adf[1])
print('p-value',IPP_tot_adf[1])
print('p-value',IPC_tot_adf[1])
##On ne peut nulle part rejeter l'hypothèse nulle de non stationnarité, il faudra donc différencier les séries avant de faire les régressions



##On régresse d'abord les coûts de l'énergie sur les prix de production globaux en stationnarisant les séries
data = pd.merge(energie_prix_import,IPP_tot, on='TIME_PERIOD', how="inner")
##On différencie les données pour travailler avec des séries stationnaires
data_diff = data.diff().dropna()
##On crée les valeurs lag des observations qui serviront de variables au modèle
lag_x = data_diff['OBS_VALUE_x'].shift(1)
lag_y = data_diff['OBS_VALUE_y'].shift(1)
data_diff['lag_x'] = lag_x
data_diff['lag_y'] = lag_y
data_diff = data_diff.dropna()
##Modelisation
X = sm.add_constant(data_diff[['lag_x','lag_y']])
y = data_diff['OBS_VALUE_y']
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
##On a déjà un effet significatif des coûts de l'énergie, à préciser l'interprétation


##Si on modélise sur période plus courte
data_sub = data[data.index>"2020-01-31"]
##On différencie les données pour travailler avec des séries stationnaires
data_sub_diff = data_sub.diff().dropna()
##On crée les valeurs lag des observations qui serviront de variables au modèle
lag_x = data_sub_diff['OBS_VALUE_x'].shift(1)
lag_y = data_sub_diff['OBS_VALUE_y'].shift(1)
data_sub_diff['lag_x'] = lag_x
data_sub_diff['lag_y'] = lag_y
data_sub_diff = data_sub_diff.dropna()
##Modelisation
X = sm.add_constant(data_sub_diff[['lag_x','lag_y']])
y = data_sub_diff['OBS_VALUE_y']
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

##Test de la relation de co-intégration
from statsmodels.tsa.api import VAR, coint
data = pd.merge(energie_prix_import,IPP_tot, on='TIME_PERIOD', how="inner")
cointegration_test = coint(data['OBS_VALUE_x'],data['OBS_VALUE_y'])
print('p-value pour test de cointegration',cointegration_test[1])
##p-valeur = 0.41 Donc il existe bien une relation de co-intégration entre nos variables


##Raisonnement économétrique menant aux modèles ECM
data = pd.merge(energie_prix_import,IPP_tot, on='TIME_PERIOD', how="inner")
##Simple régression OLS
X = sm.add_constant(data[['OBS_VALUE_x']])
y = data['OBS_VALUE_y']
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
##Comme la variable endogène n'est pas présente avec retard, on peut tester l'autocorrélation des résidus directement avec le test de Durbin-Watson
##le summary nous donne une valeur de 0.049, or pour que les résidus ne soient pas auto-corrélés à l'ordre 1, DW doit être proche de 2 (en pratique
##on accepte des modèle avec DW entre 1.5 et 2.5
##pour afficher directement la stat DW : 
from statsmodels.stats.stattools import durbin_watson
DW_stat = durbin_watson(results.resid)
print("La statistique de test de Durbin Watson a pour valeur : ",DW_stat)
##test d'homoscédasticité des résidus par le test de White
from statsmodels.stats.diagnostic import het_white
white_test = het_white(results.resid,X)
print("p-valeur du test de White : ",white_test[1])
##On rejette l'hypothèse d'homoscédasticité des résidus (p-valeur < 5%)
##Test de normalité des résidus (test de Jarque-Bera)
from scipy.stats import jarque_bera
jb_stat, p_value_JB = jarque_bera(results.resid)
print("p-valeur du test de Jarque-Bera : ",p_value_JB)
##On rejette l'hypothèse nulle de normalité des résidus

##Donc une régression simple n'est pas interprétable dans ces conditions
##La procédure de Cochrane-Orcutt peut être utilisée pour corriger de l'autocorrélation des erreurs
##Elle calcule l'autocorrélation empirique des résidus (rho) puis crée les variables Yt-rhoY(t-1) et idem pour Xt et 
##calcule l'autocrrélation des résidus de cette nouvelle régression etc jusqu'à ce qu'on obtienne une autocorrélation nulle
##Assure plus de confiance dans les test de Student et la significativité statistique des coefficients
from statsmodels.tsa.stattools import acf
acf_resid = acf(results.resid)
weights = -acf_resid[1:]
modele_cochrane_orcutt = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':12, 'weights':weights})
print(modele_cochrane_orcutt.summary())

##Test de rupture? Lorsque la plage de données est grande, il est possible que le phénomène économique d'intérêt ait subi des altérations et des ruptures.
##Le test de Chow peut être utilisé pour détecter des changements strucutrels dans les coefficients d'un modèle de régression
from statsmodels.stats.diagnostic import linear_harvey_collier
data["groupe"] = data.index>"2018-01-01"
data['interaction'] = data["groupe"]*data['OBS_VALUE_x']
#Modele avec interaction
modele_interaction = sm.OLS(data['OBS_VALUE_y'], sm.add_constant(data[['OBS_VALUE_x','interaction']])).fit()
modele_ss_interaction = sm.OLS(data['OBS_VALUE_y'], sm.add_constant(data[['OBS_VALUE_x']])).fit()
chow_test = modele_ss_interaction.compare_f_test(modele_interaction)
print(chow_test)
##Problème d'hétéroscédasticité et d'autocorrélation dans les résidus, ne donne pas de résultats utilisables
##Valeur de stat de 68.4, donc on s'attend tout de même à un rejet de l'hypothèse nulle



##Modele à Correction d'Erreur
##On a déjà validé l'hypothèse d'une relation de co-intégration, il faudrait vérifier que les séries considérées sont intégrées d'ordre 1
##Test de stationnarité des séries temporelles
data = pd.merge(energie_prix_import,IPP_tot, on='TIME_PERIOD', how="inner")
data_diff = data.diff().dropna()
energy_adf = adfuller(data_diff['OBS_VALUE_x'])
IPP_adf = adfuller(data_diff['OBS_VALUE_y'])
print('p-value',energy_adf[1])
print('p-value',IPP_adf[1])
##Parfait, les test de stationnarité rejettent l'hypothèse nulle de non stationanrité des séries utilisées
##Donc les hypothèses nécessaires pour mettre en place un modèle ECM sont bien vérifiées
##Equation de long terme
long_terme = sm.OLS(data['OBS_VALUE_y'], sm.add_constant(data[['OBS_VALUE_x']])).fit()
print(long_terme.summary())
residus = long_terme.resid
##Equation de court terme
residus_lag = pd.Series(residus.shift(1).dropna(), name='residus')
data_courte = pd.merge(data_diff,residus_lag, on='TIME_PERIOD', how="inner")
court_terme = sm.OLS(data_courte['OBS_VALUE_y'], sm.add_constant(data_courte[['OBS_VALUE_x','residus']])).fit()
print(court_terme.summary())
##Le coefficient devant residus peut s'interpréter comme une vitesse de convergence, il est bien négatif comme attendu
##Ici il vaut 1,4%, ce qui signifie qu'à chaque mois, la variable y comble 1,4% de l'écart à sa valeur de long terme (attention ici on a peu de variables explicatives)
##En l'absence de chocs consécutifs, il faudrait donc 100/1.4 = 71 mois soit environ 6 ans pour un retour à l'équilibre

##test de la stationnarité du résidu estimé en première étape
res_adf = adfuller(residus)
print('p-value',res_adf[1])
##Problème, on a un résidu non stationnaire bien que les séries régressées soient stationnaires...
##Attention, dans ce contexte de modèle ECM, aucune loi n'est connue pour les estimateurs, il est donc impossible d'interpréter les tests statistiques sur les coefficients



##Pour limiter le biais introduit à la première étape, Stock et Watson propose des dynamic OLS, où on introduit symétriquement des variations avance et retard dans la première étape
data = pd.merge(energie_prix_import,IPP_tot, on='TIME_PERIOD', how="inner")
nombre_retard = 2
for i in range(1,nombre_retard+1):
    data[f'Delta_X_lag_{i}'] = data['OBS_VALUE_x']-data['OBS_VALUE_x'].shift(i)
    data[f'Delta_X_lead_{i}'] = data['OBS_VALUE_x']-data['OBS_VALUE_x'].shift(-i)
data = data.dropna()

##Equation de long terme
long_terme = sm.OLS(data['OBS_VALUE_y'], sm.add_constant(data[['OBS_VALUE_x','Delta_X_lag_1','Delta_X_lag_2','Delta_X_lead_1','Delta_X_lead_2']])).fit()
print(long_terme.summary())
residus = long_terme.resid
##test de la stationnarité du résidu estimé en première étape
res_adf = adfuller(residus)
print('p-value',res_adf[1])
##Toujours pas stationnaire

##Equation de court terme
##Attention il faut un peu transformer le résidu avant de l'introduire dans l'équation de court terme
residus_trans = data['OBS_VALUE_y']-long_terme.params['OBS_VALUE_x']*data['OBS_VALUE_x']-long_terme.params['const']
residus_lag = pd.Series(residus_trans.shift(1).dropna(), name='residus')
data_courte = pd.merge(data_diff,residus_lag, on='TIME_PERIOD', how="inner")
court_terme = sm.OLS(data_courte['OBS_VALUE_y'], sm.add_constant(data_courte[['OBS_VALUE_x','residus']])).fit()
print(court_terme.summary())
##On passe de 1,4% à 1,8% pour la force de rappel, pas de grande différence