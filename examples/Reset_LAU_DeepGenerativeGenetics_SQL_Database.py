import requests
import argparse

def warning_message(message, end = '\n'):
	print('[WARNING] {}'.format(message), end = end)

def reset_full_sql():
	resetDatabase = 'http://LAUDeepGenerativeGenetics.pythonanywhere.com/'
	resetDatabase = resetDatabase + 'Reset'
	req = requests.get(resetDatabase)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--code', type=int)
	clargs = parser.parse_args()
	
	if clargs.code == 11235813213455: 
		warning_message('RESETTING SQL DATABASE ')
		reset_full_sql()
	else:
		warning_message("Nuh! uh! uh! You didn't say the magic word!")