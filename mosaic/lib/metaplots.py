import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import seaborn as sns

# FIXME Récupérer le bon fichier 

if __name__ == '__main__':
	db_con = sqlite3.connect('run_only_sin2.db')
	db_cur = db_con.cursor()

	df = pd.DataFrame(db_cur.execute(
		'''
		SELECT p0.run_id, p0.param_value, p1.nb_params, p1.test_loss, p2.param_name, p2.param_value
		FROM params p0
			INNER JOIN run_results p1 ON p0.run_id = p1.run_id
			INNER JOIN params p2 ON p0.run_id = p2.run_id
		WHERE p0.param_name="dataset_key" AND (p2.param_name="fitter_key")
		'''
	).fetchall(), columns=['run_id', 'data_name', 'nb_params', 'test_loss', 'key_name', 'key_value'])

	dfs = dict(tuple(df.groupby('data_name')))
	dfs = {key:dict(tuple(dfs[key].groupby('key_value'))) for key in dfs}

	for data_name in dfs:
		for model_name in dfs[data_name]:
			sns.violinplot(x=dfs[data_name][model_name]['test_loss'].values)
			plt.show()
	
	
	for key, value in dfs['2d_sin2'].items():
		print(key, value, end='\n\n')

	import matplotlib.backends.backend_pdf
	pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

	for data_name, d in dfs.items():
		fig = plt.figure()
		plt.title(data_name)			
		for model_name in d:
			plt.plot(dfs[data_name][model_name]['nb_params'], dfs[data_name][model_name]['test_loss'], label=model_name)
		plt.xscale('log')
		#plt.legend()
		pdf.savefig(fig)
		plt.close()

	pdf.close()

"""
'''
	TODO: Metaplots
	last loss / # params


	BD:

	id	|	param_name	|	param_value

'''

def regexp(pattern, string):
	print(re.search(pattern, string))
	return re.search(pattern, string)

db_con = sqlite3.connect('run_qml_full.db')
db_cur = db_con.cursor()

def fetch_distinct_values(key):
	request = '''SELECT DISTINCT param_value FROM params WHERE param_name=?'''
	db_cur.execute(request, (key, ))
	return db_cur.fetchall()

def fetch_values(filters):

	valid_operators = ['=', '>=', '<=', '<', '>', '!=']

	db_con.row_factory = lambda cursor, row:{list(filters.keys())[i-1]:row[i] for i in range(1, len(row))}
	db_cur = db_con.cursor()

	_select = "SELECT p0.run_id"
	for i in range(len(filters)):
		_select += f", p{i}.param_value"

	_from = "\nFROM params p0\n"
	_inner_join = ""
	for i in range(len(filters) - 1):
		_inner_join += f"\tINNER JOIN params p{i + 1} ON p0.run_id=p{i + 1}.run_id\n"

	_where = "WHERE"
	for i, (key, value) in enumerate(filters.items()):
		operator = '='
		if type(value) == tuple:
			operator, value = value
			if operator not in valid_operators:
				raise Exception(f'Unknown operator {operator}')

		if value == '':
			_where += f" p{i}.param_name='{key}'"
		else:
			if type(value) == str:
				_where += f" p{i}.param_name='{key}' AND p{i}.param_value='{value}'"
			elif type(value) == int or type(value) == float:
				_where += f" p{i}.param_name='{key}' AND p{i}.param_value{operator}{value}"

		if i < len(filters) - 1:
			_where += " AND"
	
	db_cur.execute(_select + _from + _inner_join + _where)
	return db_cur.fetchall()



# Récupérer tous les noms uniques

# [
# 	[{'Challenges_data_name': 'rastrigin2', 'nb_params': 6, 'test_loss': 0.6109429001808167}, {'Challenges_data_name': 'rastrigin2', 'nb_params': 6, 'test_loss': 0.6109429001808167}]
# 	[{'Challenges_data_name': 'rastrigin2', 'nb_params': 6, 'test_loss': 0.6109429001808167}, {'Challenges_data_name': 'rastrigin2', 'nb_params': 6, 'test_loss': 0.6109429001808167}]
# ]

unique_names = fetch_distinct_values('Challenges_data_name')
print(unique_names)
for name in unique_names:
	filters = {'Challenges_data_name':name[0], 'nb_params':('=', 6), 'test_loss':''} # n data
	selected_values = fetch_values(filters)
	print(selected_values[0])
"""
