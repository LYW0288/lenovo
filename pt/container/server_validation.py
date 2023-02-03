#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
import gc
import os
import time
import ctypes
import socket
import psutil
import logging
import argparse
import threading
import traceback
import subprocess
from queue import Queue, Empty
import cauculateAP as cAP
import sys


'''	____________________________________________________
	________________WHAT YOU NEED INPUT_________________
	1.	你的model位置
	2.	要檢驗的圖片位置
	3.	是否需要cauculate(取決於有沒有labeled)
	4.	log的位置
	====================================================
'''
def parse_opt():
	parser = argparse.ArgumentParser()
	parent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.path.pardir)
	
	print(parent_path)
	parser.add_argument('--model_path', type=str, default='pcba', help='where is model.')
	parser.add_argument('--val_dir', type=str, default=f'{os.path.abspath(parent_path)}/vallidation', help='where is pic for validation.')
	parser.add_argument('--cauculate', type=int, default=False, help='local or AWS.')
	parser.add_argument('--log_file', type=str, default=f'{os.path.abspath(parent_path)}/log.log', help='logfile path.')
	parser.add_argument('--out_path', type=str, default='./', help='where to out result.')
	opt = parser.parse_args()
	return opt.model_path, opt.val_dir, opt.cauculate, opt.log_file, opt.out_path

class thread_with_exception(threading.Thread): 
	def __init__(self, name, Process, args):
		threading.Thread.__init__(self)
		self.daemon = True # thread dies with the program
		self.name = name
		self.Process = Process
		self.args = args
		self.exitcode = 0
		self.exception = None

	def run(self):
		try: 
			self.Process(*self.args)
		except Exception as e:
			self.exitcode = 1
			self.exception = e
			print(traceback.format_exc())
			print(f'\033[91mthread-{self.get_id()} failed\033[0m') 
		else:
			print(f'thread-{self.get_id()} end')
		
	def get_id(self): 
		# returns id of the respective thread 
		if hasattr(self, '_thread_id'): 
			return self._thread_id 
		for id, thread in threading._active.items(): 
			if thread is self: 
				return id

	def raise_exception(self): 
		thread_id = self.get_id() 
		res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 
			ctypes.py_object(SystemExit)) 
		if res > 1: 
			ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0) 
			print('Exception raise failure') 


def exitClean(cauculate, sub_id):
	if cauculate and psutil.pid_exists(sub_id):
		killProcess(sub_id)


def enqueue_output(out, queue):
	for line in iter(out.readline, b''):
		queue.put(line)
	#out.close()

def Get_local_ip(isLocal = False):
	if isLocal:
		return "127.0.0.1"
	try:
		csock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		csock.connect(('8.8.8.8', 80))
		(addr, port) = csock.getsockname()
		csock.close()
		return addr
	except socket.error:
		return "127.0.0.1"


def killProcess(parent_pid):
	parent = psutil.Process(parent_pid)
	for child in parent.children(recursive=True):  # or parent.children() for recursive=False
		child.kill()
	parent.kill()
	
def serverValidation(model_path, val_dir, cauculate, log_file, out_path):
	logging.basicConfig(	level=logging.INFO,
							format='%(asctime)s %(message)s',
							filename=log_file,
							filemode='a')
	print(f'now thread pid is {os.getpid()}')
	now_path = os.path.dirname(os.path.abspath(__file__))
	print(now_path)
	bashCMD = [	'python', f'{now_path}/ObjectDetectionFastAPI.py', 
				'--weights', model_path,
				'--islocal', 'True']
	print(f'# of pids is {len(psutil.pids())}')
	sub = subprocess.Popen(	bashCMD, 
				stdout = subprocess.PIPE,
				stderr = subprocess.STDOUT)

	sub_id = sub.pid
	print(f'# of pids is {len(psutil.pids())}')
	print(f'_____________________\n\tPID:\t\033[1;36m{sub_id}\033[0m')
	print(f'\tPPID:\t\033[1;36m{psutil.Process(sub_id).ppid()}\033[0m')
	#print(f'[MEM]\t{psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024}')

	while sub.poll() is None:
		output = sub.stdout.readline().decode('utf-8', 'ignore')
		print(output, end="")
		if 'Application startup complete' in output:
			print('\033[1;92m')
			print('*==================*')
			print(' success connected!')
			print('*==================*','\033[0m')
			break

	thread = thread_with_exception(	name = 'cauculateAP', Process = cAP.cauculateAP,
									args = (val_dir, Get_local_ip(True), 
									out_path, cauculate))
	thread.start()
	thid = thread.get_id()
	q = Queue()
	monitor = thread_with_exception(name = 'monitor', Process = enqueue_output, 
					args = (sub.stdout, q))
	monitor.start()
	mnid = monitor.get_id()

	exitcode = 0
	exception = None
	print(f'_______________________________________')
	print(f'\tCAUCULATED TID:\t\033[1;36m{thid}\033[0m')
	print(f'\tMONITORING TID:\t\033[1;36m{mnid}\033[0m')
	print(f'thread generated.')

	while sub.poll() is None and thread.is_alive():
		if thread.exitcode:	
			(exitcode, exception) = (thread.exitcode, thread.exception)
			print('thread error')
			break
		try: output = q.get_nowait().decode('utf-8', 'ignore')
		except Empty:
			continue
		msg = f'\033[1;34m{output}\033[0m'
		print(msg, end="")
	print(f'# of pids is {len(psutil.pids())}')
	if thread.is_alive():	thread.raise_exception() 
	if monitor.is_alive():	monitor.raise_exception() 
	if psutil.pid_exists(sub_id):
		killProcess(sub_id)
		print(f'# of pids is {len(psutil.pids())}')
		print('\033[91m', 'subprocess killed','\033[0m')
	del q
	gc.collect()
	thread.join()
	monitor.join()


	parent_path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))
	if cauculate:
		with open(f'{parent_path}/monitor/status.txt', 'w+') as f:
			f.write('empty\n')
			print('empty!')
	if exitcode:
		logging.exception(exception)
		raise Exception(exception)

if __name__ == "__main__":
	model_path, val_dir, cauculate, log_file, out_path = parse_opt()
	if not os.path.isdir(out_path):
		os.makedirs(out_path)
	if not os.path.isfile(model_path):
		print(model_path)
		print('model doesnt exist!')
		parent_path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))
		if cauculate:
			with open(f'{parent_path}/monitor/status.txt', 'w+') as f:
				f.write('empty\n')
				print('empty!')
		sys.exit(1)
	serverValidation(model_path, val_dir, cauculate, log_file, out_path)