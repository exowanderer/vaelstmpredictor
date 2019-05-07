from functools import partial
from paramiko import SSHClient, AutoAddPolicy
from multiprocessing import Queue, Process
from os import environ

def debug_message(message): print('[DEBUG] {}'.format(message))
def info_message(message): info_message('{}'.format(message))

def update_all_git():
	machines = [['172.16.50.187',  'vaelstmpredictor/'],
				 ['172.16.50.181', 'vaelstmpredictor/'],
				 ['172.16.50.176', 'vaelstmpredictor/'],
				 ['172.16.50.177', 'vaelstmpredictor/'],
				 ['172.16.50.163', 'vaelstmpredictor/'],
				 # ['172.16.50.182', 'vaelstmpredictor/'],
				 ['172.16.50.218', 'vaelstmpredictor/'],
				 ['172.16.50.159', 'vaelstmpredictor/'],
				 ['172.16.50.235', 'vaelstmpredictor/'],
				 ['172.16.50.157', 'vaelstmpredictor/'],
				 ['172.16.50.237',  'vaelstmpredictor/']]
	
	for hostname, basedir in machines:
		partial_update = partial(update_one_git, 
					hostname = hostname, basedir = basedir)
		process = Process(target=partial_update)
		process.start()

def update_one_git(hostname, username = "acc", basedir = 'vaelstmpredictor/',
					port = 22, verbose = True, private_key='id_ecdsa'):
	
	key_filename = environ['HOME'] + '/.ssh/{}'.format(private_key)
	
	ssh = SSHClient()
	ssh.set_missing_host_key_policy(AutoAddPolicy())
	ssh.connect(hostname, key_filename = key_filename)

	command = []
	command.append('cd {}'.format(basedir))
	command.append('git branch')
	command.append('git remote get-url origin')
	command.append('git pull')
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
		print('error on stderr.readlines(): {}'.format(str(e)))

	print("Command Executed Successfully")
	ssh.close()

if __name__ == '__main__':
	update_all_git()