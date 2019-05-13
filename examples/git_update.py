#! /home/acc/anaconda3/envs/tf_gpu/bin/python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--commit_message', nargs='?')

clargs = parser.parse_args()

import subprocess
import os
subprocess.run("ls -l".split(' '))
subprocess.call("ls -l".split(' '))

os.chdir('/home/acc/github/vaelstmpredictor/')

subprocess.run("git status".split(' '))
subprocess.run("git add -u".split(' '))
subprocess.run("git commit -m {}".format(clargs.commit_message).split(' '))
subprocess.run("git push".split(' '))
subprocess.run("python update_machines_via_github.py".split(' '))

os.chdir('/home/acc/github/vaelstmpredictor/examples')

print('[INFO] Completed Subprocess')