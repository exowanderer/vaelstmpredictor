import pandas as pd
import requests

from argparse import ArgumentParser
from time import time

def info_message(message, end='\n'):
	print('[INFO] {}'.format(message), end = end)

def save_sql_to_csv(table_dir, hostname='172.16.50.176', sqlport=5000):
	getDatabase = 'http://{}:{}/GetDatabase'.format(hostname, sqlport)

	info_message('Accessing SQL from {}'.format(getDatabase))

	req = requests.get(getDatabase)
	sql_table = pd.DataFrame(req.json())

	time_stamp = sql_table['time_stamp'][0]
	run_name = sql_table['run_name'][0]

	info_message('Found table with time_stamp SQL {}'.format(time_stamp))
	
	table_dir = clargs.table_dir
	table_name = '{}/{}_fitness_table_{}.csv'
	table_name = table_name.format(table_dir, run_name, time_stamp)

	info_message('Storing table to {}'.format(table_name))

	sql_table.to_csv(table_name)

parser = ArgumentParser()
parser.add_argument('-t', '--table_dir', type=str, default='../data/tables',
	help='/path/to/and/name of the table to store the sql database')
parser.add_argument('-h', '--hostname', type=str, default='127.0.0.1',
	help='/path/to/and/name of the table to store the sql database')
parser.add_argument('-p', '--sqlport', type=int, default=5000,
	help='/path/to/and/name of the table to store the sql database')

clargs = parser.parse_args()

save_sql_to_csv(clargs.table_dir, clargs.hostname, clargs.sqlport)