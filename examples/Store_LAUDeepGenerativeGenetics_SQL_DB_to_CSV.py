import pandas as pd
import requests

from argparse import ArgumentParser
from time import time

def info_message(message, end='\n'):
	print('[INFO] {}'.format(message), end = end)

def save_sql_to_csv(table_dir, 
				hostname = 'LAUDeepGenerativeGenetics.pythonanywhere.com', 
				sqlport = 5000):
	
	# getDatabase = 'http://{}:{}/GetDatabase'.format(hostname, sqlport)
	getDatabase = 'http://{}/GetDatabase'.format(hostname, sqlport)

	info_message('Accessing SQL from {}'.format(getDatabase))

	req = requests.get(getDatabase)
	sql_table = pd.DataFrame(req.json())
	if len(sql_table) == 0:
		info_message('No Database To Save at {}'.format(getDatabase))
		return

	time_stamp = sql_table['time_stamp'][0]
	run_name = sql_table['run_name'][0]

	info_message('Found table with time_stamp SQL {}'.format(time_stamp))
	
	table_dir = clargs.table_dir
	table_name = '{}/{}_fitness_table_{}.csv'
	table_name = table_name.format(table_dir, run_name, time_stamp)

	info_message('Storing table to {}'.format(table_name))

	sql_table.to_csv(table_name)

parser = ArgumentParser()
parser.add_argument('--table_dir', type=str, default='../data/tables',
	help = '/path/to/and/name of the table to store the sql database')
parser.add_argument('--sqlhost', type=str, 
	default = 'LAUDeepGenerativeGenetics.pythonanywhere.com',
	help = '/path/to/and/name of the table to store the sql database')
parser.add_argument('--sqlport', type=int, default=5000,
	help = '/path/to/and/name of the table to store the sql database')

clargs = parser.parse_args()

save_sql_to_csv(clargs.table_dir, clargs.sqlhost, clargs.sqlport)