# Tutorial

Disons que nous voulons comparer deux architectures de MLP sur un dataset d'opérateur logique OR. Grâce à Mosaic, il est possible de combiner des datasets et des modèles pour faire des pipelines d'entrainement et ainsi analyser deux modèles sur un même dataset. Créez vos classes, écrivez un fichier de configuration et lancez les runs.

Toutes les données issues de l'entrainement sont stockées dans une base de donnée et peuvent être analysées grâce aux commandes `mosaic_plotloss` `mosaic_metaplot`.


### Step 1
Commençons par créer nos classes de dataset.

Nous devons implémenter la méthode prepare. Méthode qui sera appelé lors de l'execution de la pipeline pour toutes les classes qui se trouvent dans l'argument data_scheme du fichier de configuration. C'est dans cette méthode que vous allez charger des données de fichiers ou en générer des nouvelles. Le framework attendra du dernier mondule des classes qui se trouvent dans la catégorie data_scheme qu'elle renvoie deux DataLoader. Un pour le jeu d'entrainement et l'autre pour le jeu de test.
Nous avons choisi d'utiliser pytorch pour la gestion de donnée du framework. Il est donc nécessaire de renvoyer des DataLoader pytorch.

Nous devons implémenter la méthode info dans laquelle nous renvoyons le paramètre batch_size qui se trouvera dans le fichier de configuration dans la section associée au dataset et toutes les autres valeurs que nous souhaitons stocker pour la suite de l'execution de la pipeline. Dans notre cas, la taille d'entrée et de sortie du MLP nous seront utiles.

On pourra ainsi faire une méthode qui ressemble à :
```python
def info(self):
	return {'batch_size' : self.batch_size, 'input_size' : 2, 'output_size' : 1}
```

### Step 2

Implementons maintenant les deux classes de mlp que nous voulons comparer. On peut par exemple changer la forme du mlp et faire une forme de brique et une forme d'entonoir pour voir si l'un ou l'autre à de meilleurs résultats.
Le type des deux mlp dans le fichier de configuration devra être le même et devra être spécifié dans l'argument pipeline_scheme de la section `PROCESS`.

La forme du mlp sera défini dans l'initalisation de la classe et les paramètre de la fonction \_\_init__ doivent avoir le même nom que ceux présents dans le fichier de configuration ou celui des clés du dictionnaire renvoyé par les méthodes info().

Dans notre cas, le prototype de la fonction \_\_init__ de nos mlp sera:
```python
# input_size et output_size viennent de la méthode info() du dataset
# length et width viennent du fichier de configuration
def __init__(self, input_size, output_size, length, width)
```

Dans chacuns d'entre eux, nous devons implémenter la méthode forward qui sera éxecutée lors de l'execution pour transformer les données.
Nous devons également implémenter la méthode info qui renverra un dictionnaire d'informations utiles aux prochains modules dans la pipeline. Etant donné que nous avons qu'un seul module dans notre pipeline, nous renverrons un dictionnaire vide.

Si les classes qui se trouvent dans le pipeline_scheme ne possède par de méthode parameters() par default (héritage du Module de pytorch) il faudra alors l'implémenter pour renvoyer les paramètres du modèle. Dans notre cas nous feront hériter nos classes Mlp de *torch.nn.Module*, nous avons donc pas besoin d'implémenter la méthode parameters().

Enfin, pour sauvegarder et charger le modèle, il faut implémenter la méthode save_model et load_model. Choisissez la méthode que vous voulez pour sauvegarder du moment que load_model sait comment récupérer les informations. Le chemin spécifié dans les arguments ne possède pas d'extension, il faut donc rajouter celle avec laquelle nous voulons sauvegarder ou charger le modèle.

### Step 3

Passons maintenant au fichier de configuration.

Ajoutons dans un premier temps les deux sections obligatoires: `PROCESS` et `MONITOR`
qui possèdent des arguments de gestion de la méthode d'execution ou des informations relatives à tous les runs.

Ensuite ajoutons pour chaque classes que l'on veut comparer, sa section avec les arguments que l'on souhaite lui faire passer.
Nous devons spécifier les arguments suivants obligatoirement: type, classe, path_to_class, key
Dans notre cas, nous voulons rajouter une pronfondeur et une largeur pour notre architecture de mlp donc nous l'indiquerons dans leur section du fichier de configuration.

key permet un requêtage simplifié de la base de donnée. Il est donc important de la choisir comme il faut. Si elle ne vous convient pas au moment de faire des plots, vous pouvez mettre à jour votre base de donnée en utilisant `mosaic_rekey` après avoir regénéré les pipelines concernées grâce à `mosaic_generate`.

### Step 4

Nous avons désormais tous les pré-requis pour lancer les entrainements.

La commande `mosaic_run` permet de lancer l'execution des pipelines grâce au fichier de configuration qu'on lui passe en paramètre

A tout moment, il est possible de faire pause avec `mosaic_pause`, de faire une sauvegarde de la base de donnée avec `mosaic_savedb` ou simplement suivre l'état de l'execution avec `mosaic_status`.

Si un problème est apparu ou que vous aimeriez continuer l'entrainement, il est possible de relancer qu'une partie des pipelines avec `mosaic_rerun`.

### Step 5

Enfin, pour analyser les résultats, deux commandes sont disponibles: `mosaic_plotloss` qui vous permettra de comparer les courbes d'apprentissage des différents modèles et `mosaic_metaplot` qui vous permettra de comparer certains paramètres comme le taux d'overfitting en fonction du nombre de paramètre pour un dataset donné