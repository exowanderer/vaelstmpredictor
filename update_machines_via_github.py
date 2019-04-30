from functools import partial
from paramiko import SSHClient, AutoAddPolicy
from multiprocessing import Queue, Process
from os import environ

def update_all_git():
	hostnames = ['172.16.50.181',
				'172.16.50.176',
				'172.16.50.177',
				'172.16.50.163',
				'172.16.50.182',
				'172.16.50.218',
				'172.16.50.159',
				'172.16.50.235',
				'172.16.50.157',
				'172.16.50.237']
	
	for hostname in hostnames:
		partial_update = partial(update_one_git, hostname = hostname)
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

	print('[INFO] Executing {} on {}'.format(command, hostname))

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
	
	print("Command Executed")
	ssh.close()

if __name__ == '__main__':
	update_all_git()