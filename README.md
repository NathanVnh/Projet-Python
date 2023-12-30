# Projet Python

Le notebook à lancer se nomme execute.

**Dépôt GitHub pour le Projet de Python pour la data-science (ENSAE)**

L'idée de ce projet est de lister un ensemble non exhaustif d'outils pour analyser l'inflation à partir de séries macroéconomiques provenant de l'Insee et d'Eurostat. Nous montrerons que l'idée naïve d'utiliser l'outil de base des statisticiens qu'est la régressions des moindres carrés ordinaires rend les coefficients et les tests difficilement interprétables. Nous introduirons, au moyen de 2 exemples, une méthode de modélisation qui permet de corriger cela.


## Récupération des données
Les données sont récupérées via l'API BDM de l'Insee en utilisant le package *pynsee*, cette opération nécessite l'initalisation d'une clé et d'un code de sécurité qui sont ensuite transmissibles à d'autres utilisateurs. Ainsi ce code est lançable depuis n'importe quelle machine sans nécessité au préalable d'avoir créé un compte sur les API Insee. On se sert ici des Idbank des séries qui peuvent être identifiées au préalable pour vos séries d'intérêt directement via le site de l'Insee https://www.insee.fr/fr/information/3128533
Un autre type de données considérée sont les données issues d'eurostats (https://ec.europa.eu/eurostat/fr/web/hicp/database), notamment concernant l'IPCH, l'indice des prix à la consommation harmonisés et de ses diverses composantes, par exemple le prix des fruits. 

## Programmation orientée objet
Nous nous intéresserons plus à l'importation, la visualisation et la modélisation des données qu'à l'interprétation proprement socioéconomique qu'on peut avoir de ces résultats. Pour ce faire, nous avons réalisé 2 classes, chacune utilisée en fonction de l'origine des données (Insee ou Eurostats). A travers ces classes, des attributs et méthodes ont été codéss afin de pouvoir généraliser les analyses que nous allons faire sur n'importe quelles données issues de ces 2 bases de données. Ainsi, des méthodes ont été codées pour réaliser l'import des données, leur nettoyage ainsi que différentes visualisations et modélisations possible à partir de ces séries temporelles.

## Méthodologie pour l'analyse des déterminants macroéconomiques de l'inflation
On va s'intéresser ici à des Séries Temporelles, il est donc possible en première analyse de réaliser une décomposition en tendance, saisonnalité et résidus pour avoir notamment une idée des variations qu'ont pu connaître les séries d'intérêt et mettre en évidence les chocs par l'étude du processus résiduel.

### Tests statistiques
On peut tester la stationnarité des processus considérés grâce au **test augmenté de Dickey-Fuller**. La stationnarité implique que les propriétés statistiques de la série (espérance, variance, auto-corrélation) ne varient pas dans le temps. A noter que l'hyptohèse nulle est celle de non-stationnarité.

On peut tester l'autocorrélation des résidus avec le **test de Durbin-Watson**. L'hypothèse nulle stipule qu'il y a non auto-corrélation. Le test de Durbin-Watson conduit à régresser le résidu à l'instant t sur le résidu précédent au temps t-1 selon le modèle : 
$\epsilon_{t}$ = $\beta$ $\epsilon_{t-1}$ + $u_{t}$ avec $u_{t}$ un bruit blanc de régression classique et $\epsilon_{t}$ le résidu de la régression OLS pour laquelle on veut tester l'autocorrélation résiduelle.

On peut tester l'homoscédasticité des résidus par le **test de White**, d'hypothèse nulle l'homoscédasticité des résidus.

Le **test de Jarque-Bera** permet de conclure quant à la normalité du processus résiduel (donc non autocrrélation et homoscédasticité en plus), son hypothèse nulle est la distribution normale.


### Modélisation par Error Correction Model
Pour étudier les déterminants macroéconomiques de l'inflation, il convient de mettre en place des modèles de régression. Or, les formules habituelles d'économétrie impliquent qu'il faille travailler avec des séries temporelles stationnaires puisque, par exemple, la moyenne des variables est susceptible de diverger avec le nombre d'observations si la série considérée n'est pas stationnaire. Toutefois, les séries considérées en macro-économie ne sont pas toujours stationnaires (souvent au minimum intégrées d'ordre 1).
#### Notion de non stationnarité
> Une série $X_{t}$ est dite stationnaire si ses deux premiers moments, à savoir son espérance $\mu_{t}$ = E($X_{t}$) et les autocovariances $\gamma_{tk}$ = cov($X_{t}$, $X_{k}$), sont finies et indépendantes du temps.
Visuellement une telle série tend à retourner à sa moyenne quand elle s'en est écartée sous l'effet de chocs.

> La principale forme de non-stationnarité est due à la présence d'une tendance, et là deux cas sont à distinguer :
    - Il peut exister une tendance déterministe dans le processus, qui s'écrit alors sous la forme $X_t = a*t + \epsilon_t$ ; on parle de modèle trend-stationnaire.
    - Ou alors on peut observer une tendance stochastique due à la présence d'une racine unitaire, par exemple le processus $X_t = X_{t-1} + \epsilon_t$.
La différence principale entre ces deux types de tendances réside dans la persistance des chocs. Pour les trend-stationnaires, les chocs s'estompent (forme de retour à la moyenne) alors que pour les racines unitaires les chocs sont permanents et il est donc impossible de proposer des prévisions fiables.

> Il est important de souligner que les techniques générales d'inférence statistique ne fonctionnent plus de façon usuelle quand les séries considérées sont non-stationnaires. En particulier l'estimateur des coefficients est super-convergent (vitesse de convergence en 1/T au lieu de l'habituelle $\sqrt{T}$, de plus on n'a plus l'habituelle convergence de l'écart entre l'estimateur et la vraie valeur vers une loi normale). De ce fait, les tests usuels de Student ne sont plus valides puisque ce n'est plus la même loi asymptotique. On ne peut régresser une série non stationnaire sur une autre série non stationnaire que s'il existe une combinaison linéaire des deux séries qui, elle, est stationnaire (on dit alors que les séries sont co-intégrées).


Il est tout de même possible de régresser des séries non stationnaires entre elles, mais uniquement dans un contexte particulier de séries co-intégrées ; autrement dit s'il existe une combinaison linéaire de ces séries non stationnaires donnant lieu à un processus stationnaire. Cette combinaison linéaire correspond à l'équation de long terme du **modèle ECM (Error Correction Model)**. Cette équation sert à définir une relation d'équilibre. Le résidu issu de celle-ci définit donc une distance à l'équilibre. Le modèle ECM repose sur une seconde équation de court terme qui utilise les variables en variation et le résidu retardé de l'équation de long terme. Dans cette seconde équation, les séries considérées sont stationnaires, ce qui évite les problèmes de régression fallacieuse et permet de définir avec le résidu une force de rappel pour ramener le système à la situation d'équilibre. Ainsi, une modélisation ECM permet de capter les phénomènes de rattrapages ayant lieu après de fortes crises/expansions.

Afin de réduire le biais d'estimation propre à l'équation de long terme, on utilise la **méthode d'estimation de Stock et Watson** consistant à introduire des variations avancées et retardées des variables explicatives.
Pour tester l'existence d'une relation de co-intégration, on peut utiliser le test de Johansen basé sur la statistique de la trace, d'hypothèse nulle l'absence de relation de co-intégration, ou le test de co-intégration de Phillips-Ouliaris d'hypothèse nulle la non co-intégration de la matrice des variables.



## Quelques exemples
Dans le code du projet vous trouverez plusieurs exemples d'analyses utilisant la méthodologie précédemment développée.

Dans un premier temps, afin d'introduire les notions, on étudiera le lien entre la hausse des prix de l'énergie et la hausse des prix de production. On montrera qu'une simple régression par moindres carrés ordinaires ne permet pas de vérifier les tests habituels sur les résidus et on introduira la notion de co-intégration et de modélisation ECM.

Dans un second temps, on étudiera comment les prix des matières premières nécessaires à la construction automobile ont impacté le prix de production associé. Après une brève description des caractéristiques des variables utilisées, notamment au moyen d'outils graphiques, nous chercherons à montrer l'apport de l'introduction d'une équation de long terme et d'une force de rappel dans l'équation de court terme. En effet, il peut être tentant de se limiter à travailler sur les différences du premier ordre des variables considérées mais, en période de crise économique, l'idée de corriger de l'éloignement à une relation de long terme entre le prix des matières premières et le prix du produit fini a une importance toute particulière. Nous verrons donc en quoi l'introduction de cette "force de rappel" modifie et améliore la modélisation de notre équation de court terme (en différences premières).
