import rpyc

def save_path_file(path):
	con = rpyc.connect('localhost', port=1234)
	con._config['sync_request_timeout'] = None
	con.root.save_path_file(path)
	con.close()

def get_path_file(path):
	con = rpyc.connect('localhost', port=1234)
	con._config['sync_request_timeout'] = None
	ret = con.root.get_path_file(path)
	con.close()
	return ret

def download_file(url, path):
	con = rpyc.connect('localhost', port=1234)
	con._config['sync_request_timeout'] = None
	returned_path = con.root.download_file(url, path)
	con.close()
	return returned_path
