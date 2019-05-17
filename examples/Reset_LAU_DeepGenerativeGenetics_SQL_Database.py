import requests
import argparse

def warning_message(message, end = '\n'):
	print('[WARNING] {}'.format(message), end = end)

def reset_full_sql(clargs):
	# resetDatabase = 'http://LAUDeepGenerativeGenetics.pythonanywhere.com/'
	resetDatabase = 'http://{}:{}/Reset'.format(clargs.hostname, clargs.sqlport)
	req = requests.get(resetDatabase)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--code', type=int)
	parser.add_argument('--sqlport', type=int, default=5000,
			help='The port on the work computer to send ssh over.')
	parser.add_argument('--hostname', type=str, default='172.16.50.176',
			help='The port on the work computer to send ssh over.')
	clargs = parser.parse_args()
	
	if clargs.code == 11235813213455: 
		warning_message('RESETTING SQL DATABASE ')
		reset_full_sql(clargs = clargs)
	else:
		warning_message("Nuh! uh! uh! You didn't say the magic word!")