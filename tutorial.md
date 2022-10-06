# Tutorial

Disons que nous voulons comparer deux architectures de MLP sur un dataset d'opérateur logique OR. Grâce à Mosaic, il est possible de combiner des datasets et des modèles pour faire des pipelines d'entrainement et ainsi analyser deux modèles sur un même dataset. Créez vos classes, écrivez un fichier de configuration et lancez les runs.

Nous voulons comparer une architecture de mlp en forme de brique avec un mlp en forme d'entonoir pour voir à quel point la forme du mlp joue sur la performance du modèle. Dans un cas les hiden layers seront tous de la même taille alors que dans l'autre leur taille sera décroissants. Pour mieux les comparer, les deux modèles auront le même nombre de neurones.

### Step 1
Commençons par créer nos classes de dataset.

Nous devons implémenter la méthode prepare. Méthode qui sera appelé lors de l'execution de la pipeline pour toutes les classes qui se trouvent dans l'argument data_scheme du fichier de configuration. C'est dans cette méthode que vous allez charger des données de fichiers ou en générer des nouvelles. Le framework attendra du dernier mondule des classes qui se trouvent dans la catégorie data_scheme qu'elle renvoie deux DataLoader. Un pour le jeu d'entrainement et l'autre pour le jeu de test.

```python
def prepare(self, data, run_info, module_info):
	...

	train_loader = torch.utils.data.DataLoader(train_dataset,
	batch_size=self.batch_size, drop_last=True)

	test_loader = torch.utils.data.DataLoader(test_dataset,
	batch_size=self.batch_size, drop_last=True)

	return train_loader, test_loader
```
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

Dans notre cas, forward sera implementé de cette façon.
```python
def forward(self, data, run_info, module_info):
	for layer in self.layers:
		data = self.activation(layer(data))
	return data
```

Nous devons également implémenter la méthode info qui renverra un dictionnaire d'informations utiles aux prochains modules dans la pipeline. Etant donné que nous avons qu'un seul module dans notre pipeline, nous renverrons un dictionnaire vide.

Si les classes qui se trouvent dans le pipeline_scheme ne possède par de méthode parameters() par default (héritage du Module de pytorch) il faudra alors l'implémenter pour renvoyer les paramètres du modèle sous forme d'un tenseur, de la même manière que parameters des modules pytorch. Dans notre cas nous feront hériter nos classes Mlp de *torch.nn.Module*, nous avons donc pas besoin d'implémenter la méthode parameters().

Enfin, pour sauvegarder et charger le modèle, il faut implémenter la méthode save_model et load_model. Choisissez la méthode que vous voulez pour sauvegarder du moment que load_model sait comment récupérer les informations. Le chemin spécifié dans les arguments ne possède pas d'extension, il faut donc rajouter celle avec laquelle nous voulons sauvegarder ou charger le modèle.

Dans notre cas:
```python
def save_model(self, path):
	path += '.pt'
	torch.save(self, path)

def load_model(self, path):
	path += '.pt'
	self = torch.load(path)
```

### Step 3

Passons maintenant au fichier de configuration.

Ajoutons dans un premier temps les deux sections obligatoires: `PROCESS` et `MONITOR` à notre fichier config.ini

```ini
[PROCESS]
lr = 1e-2
epochs = 400
data_scheme = dataset
pipeline_scheme = mlp

[MONITOR]
need_gpu = False
gpu_available = None
nb_processus = 8
multiplicity = 4
```

qui possèdent des arguments de gestion de la méthode d'execution ou des informations relatives à tous les runs.

Ensuite ajoutons pour chaque classes que l'on veut comparer, sa section avec les arguments que l'on souhaite lui faire passer.
Nous devons spécifier les arguments suivants obligatoirement: type, classe, path_to_class, key
Par exemple, pour notre classe mlp_funnel:

```ini
[mlp_funnel]
type = mlp
class = mlp_funnel
path_to_class = ./mosaic/share/mlp.py
key = mlp_funnel
length = {2-5}
width = 3
```

key permet un requêtage simplifié de la base de donnée. Il est donc important de la choisir comme il faut. Si elle ne vous convient pas au moment de faire des plots, vous pouvez mettre à jour votre base de donnée en utilisant `mosaic_rekey` après avoir regénéré les pipelines concernées grâce à `mosaic_generate`.

### Step 4

Nous avons désormais tous les pré-requis pour lancer les entrainements.

La commande `mosaic_run` permet de lancer l'execution des pipelines grâce au fichier de configuration qu'on lui passe en paramètre

```bash
>> mosaic_run config.ini database.db
 ---> launch 1 1/24
 ---> launch 2 2/24
 ---> launch 3 3/24
 ---> launch 4 4/24
 ---> launch 5 5/24
 ---> launch 6 6/24
 ---> launch 7 7/24
 ---> launch 8 8/24
 [run finished] 1
 ---> launch 9 9/24
```
Comme il n'est pas conseillé d'intéragir avec la base de donnée pendant le run, la commande `mosaic_savedb` permet de faire une copie de la base de donnée à tout moment du run.

```bash
>> mosaic_savedb database_copy.db
```

Vous pouvez suivre l'avancement des runs en utilisant la commande `mosaic_status`.

```bash
>> mosaic_status
Done
        [1] train_loss=4.171E-11        test_loss=4.074E-11     execution_time=4.670E+01
        [2] train_loss=4.264E-11        test_loss=3.928E-11     execution_time=4.825E+01
        [3] train_loss=4.185E-11        test_loss=4.125E-11     execution_time=4.820E+01
        [4] train_loss=4.065E-11        test_loss=3.964E-11     execution_time=4.841E+01
        [6] train_loss=4.974E-11        test_loss=4.840E-11     execution_time=4.762E+01
        [7] train_loss=4.157E-11        test_loss=4.254E-11     execution_time=4.846E+01
        [8] train_loss=4.084E-11        test_loss=4.036E-11     execution_time=4.784E+01
Error
Running
        [5] dataset_OR(1,200,0.8) | mlp_funnel(4,3)
        [9] dataset_OR(1,200,0.8) | mlp_funnel(4,4)
        [10] dataset_OR(1,200,0.8) | mlp_funnel(4,4)
        [11] dataset_OR(1,200,0.8) | mlp_funnel(4,4)
        [12] dataset_OR(1,200,0.8) | mlp_funnel(4,4)
        [13] dataset_OR(1,200,0.8) | mlp_brick(4,2)
        [14] dataset_OR(1,200,0.8) | mlp_brick(4,2)
        [15] dataset_OR(1,200,0.8) | mlp_brick(4,2)
```

Les runs finis sont décris par leur id, les dernières loss atteintes et le temps d'execution. 

Chaque runs en cours est décrit par son id suivi de la pipeline formaté qui lui est associé. Le formatage est fait en fonction de l'ordre paramètre du fichier de configuration pour chaque module présent dans la pipeline.

Les erreurs sont renvoyées avec la pipeline associé et l'id du run.

Si un problème est apparu ou que vous aimeriez continuer l'entrainement, il est possible de relancer qu'une partie des pipelines avec `mosaic_rerun`.
Si je veux par exemple relancer 200 epochs sur les 3 premiers runs, je peux faire
```bash
>> mosaic_rerun config.ini database.db -id 1-3 -epochs 200
```

### Step 5

Toutes les données issues des différents runs sont stockées dans une base de donnée et peuvent être analysées grâce aux commandes `mosaic_plotloss` `mosaic_metaplot`.

```bash
>> mosaic_plotloss database.db plotloss_output.pdf -id all -plot_size 3
```

Une partie des résultats de plotloss
![plotloss](mosaic/doc/plot_loss_demo.png)


```bash
>> mosaic_metaplot database.db metaplot.png dataset_OR test_loss
```

Résultat de metaplot
- test_loss
![metaplot](mosaic/doc/metaplot_testloss_demo.png)
- overfiting
![metaplot2](mosaic/doc/metaplot_overfit_demo.png)

On remarque que comparer l'overfitting n'est pas très pertinant car les valeurs sont très petites.

Cependant, on peut quand même constater que le mlp_brick semble mieux apprendre que mlp_funnel.