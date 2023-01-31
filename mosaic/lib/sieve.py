import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def extract_columns(filters_tab1, filters_tab2, database_path):
	tab1_size = len(filters_tab1)
	_select = "SELECT DISTINCT p0.run_id"
	for i in range(tab1_size):
		_select += f", p{i}.param_value"
	
	for i in range(len(filters_tab2)):
		_select += f", p{tab1_size}.{filters_tab2[i]}"

	_from = "\nFROM params p0\n"
	_inner_join = ""
	for i in range(len(filters_tab1) - 1):
		_inner_join += f"\tINNER JOIN params p{i + 1} ON p0.run_id = p{i + 1}.run_id\n"
	if len(filters_tab2):
		_inner_join += f"\tINNER JOIN run_results p{tab1_size} ON p0.run_id = p{tab1_size}.run_id\n"
	
	_where = "WHERE"
	for i, value in enumerate(filters_tab1):
		if i != 0:
			_where += " AND"
		_where += f" p{i}.param_name='{value}'"

	sql_request = _select + _from + _inner_join + _where

	con = sqlite3.connect(database_path)
	cur = con.cursor()
	df = pd.DataFrame(cur.execute(sql_request).fetchall(), columns=['run_id'] + filters_tab1 + filters_tab2)
	return df


def apply_cuts(df, cuts):
	for col, value, operator in cuts:
		if operator == '=':
			df = df[df[col] == value]
		elif operator == '>=':
			df = df[df[col] >= value]
		elif operator == '<=':
			df = df[df[col] <= value]
		elif operator == '>':
			df = df[df[col] > value]
		elif operator == '<':
			df = df[df[col] < value]
		elif operator == '!=':
			df = df[df[col] != value]

	return df


def apply_groupy(df, key):
	dfs = dict(tuple(df.groupby(key)))
	return dfs

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('database_path', type=str)

	args = vars(parser.parse_args())
	database_path = args['database_path']

	# critère ordonnée + nb_params + dataset_key
	# Boucler sur les modèles

	df = extract_columns(['shaped_mlp_shape', 'Challenges_data_name', 'shaped_mlp_activation'], ['test_loss', 'nb_params'], database_path)
	df = apply_cuts(df, [('shaped_mlp_shape', 'brick', '='), ('Challenges_data_name', 'sin2', '='), ('shaped_mlp_activation', 'tanh', '=')])
	dfs = apply_groupy(df, 'shaped_mlp_shape')

	fig, axes = plt.subplots(2)

	df_mean = dfs['brick'].groupby('nb_params').mean().reset_index()

	sns.violinplot(ax=axes[0], data=dfs['brick'], y='test_loss', x='nb_params', cut=0)
	axes[1].plot(df_mean['nb_params'], df_mean['test_loss'])

	axes[1].set_xticks(df_mean['nb_params'])
	plt.show()