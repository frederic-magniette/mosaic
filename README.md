MOSAIC Models for AI comparator

Mosaic is a framework dedicated to the comparison of AI models. It is often very
 difficult to choose the best AI model for a specific problematic and multiple o
ptions are available, including choices over the model hyper-parameters. It is t
empting to try different options and to compare them to get the best performance
/resource ratio. But this kind of test can be pretty time consuming from the dev
elopper point of vue. The Mosaic framework eases the automation of the program g
eneration and provides tools to help the study (Database, plot system…)

Mosaic is a python framework based on Pytorch. From a simple configuration file,
 a set of pipelines is generated including all the steps of the data treatment (
data loading, formatting, normalization, post-treatment…) and the model training
 itself. The framework executes all these pipelines in a parallel way and store 
all the results in a database and in differents files. Some facility are offered
 to pause/resume and monitor the run. A plot module helps getting some compact a
nd graphical representations to ease the interpretation of this data.
