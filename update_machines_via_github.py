import argparse

from functools import partial

import warnings

with warnings.catch_warnings():
	warnings.simplefilter('ignore')
	from paramiko import SSHClient, AutoAddPolicy

warnings.filterwarnings(action='ignore',module='.*paramiko.*')

from multiprocessing import Queue, Process
from os import environ

def debug_message(message): print('[DEBUG] {}'.format(message))
def warning_message(message): print('[WARNING] {}'.format(message))
def info_message(message): print('[INFO] {}'.format(message))

def update_all_git(reinstall = False, machines = [], branchname = 'master'):
	for hostname, basedir in machines:
		partial_update = partial(update_one_git, 
					hostname = hostname, basedir = basedir,
					reinstall = reinstall, branchname=branchname)
		process = Process(target=partial_update)
		process.start()

def update_one_git(hostname, username = "acc", basedir = 'vaelstmpredictor/',
					branchname = 'master', port = 22, verbose = True, 
					private_key = 'id_ecdsa', reinstall = False):
	
	key_filename = environ['HOME'] + '/.ssh/{}'.format(private_key)
	
	ssh = SSHClient()
	ssh.set_missing_host_key_policy(AutoAddPolicy())
	ssh.connect(hostname, key_filename = key_filename)
	
	command = []
	command.append('cd {}'.format(basedir))
	command.append('git branch')
	command.append('git remote get-url origin')
	command.append('git checkout {}'.format(branchname))
	command.append('git pull')

	if reinstall:
		setup_py = environ['HOME']+'/anaconda3/envs/tf_env/bin/python setup.py'
		command.append(setup_py + ' install')
		command.append(setup_py + ' develop')

	command = '; '.join(command)

	info_message('Executing {} on {}'.format(command, hostname))

	stdin, stdout, stderr = ssh.exec_command(command)
	
	try:
		stdout.channel.recv_exit_status()
		for line in stdout.readlines(): print(line)
	except Exception as e:
		print('error on stdout.readlines(): {}'.format(str(e)))

	try:
		stderr.channel.recv_exit_status()
		for line in stderr.readlines(): print(line)
	except Exception as e:
		warning_message('error on stderr.readlines(): {}'.format(str(e)))

	ssh.close()

	info_message('Command Executed Successfully on {}'.format(hostname))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--reinstall', action="store_true",
				help="Install vaelstmpredictor under the tf_env environment")
	parser.add_argument('--all_machines', action="store_true",
				help="Copy/Install vaelstmpredictor to all machines")
	parser.add_argument('--branchname', type=str, default='conv1d_model',
				help="name of git branch to checkout")
	clargs = parser.parse_args()

	machines = [['172.16.50.187',  'vaelstmpredictor/'],
				 ['172.16.50.181', 'vaelstmpredictor/'],
				 ['172.16.50.176', 'vaelstmpredictor/'],
				 ['172.16.50.177', 'vaelstmpredictor/'],
				 ['172.16.50.163', 'vaelstmpredictor/'],
				 ['172.16.50.182', 'vaelstmpredictor/'],
				 ['172.16.50.218', 'vaelstmpredictor/'],
				 ['172.16.50.159', 'vaelstmpredictor/'],
				 ['172.16.50.235', 'vaelstmpredictor/'],
				 ['172.16.50.157', 'vaelstmpredictor/'],
				 ['172.16.50.237',  'vaelstmpredictor/']]

	if clargs.all_machines:
		machines.extend(
				[#['172.16.50.142',  'vaelstmpredictor/'],
				 ['172.16.50.183', 'vaelstmpredictor/'],
				 ['172.16.50.184', 'vaelstmpredictor/'],
				 ['172.16.50.185', 'vaelstmpredictor/'],
				 ['172.16.50.186', 'vaelstmpredictor/'],
				 ['172.16.50.236', 'vaelstmpredictor/']])
	
	update_all_git(reinstall=clargs.reinstall, 
					machines = machines,
					branchname=clargs.branchname)
