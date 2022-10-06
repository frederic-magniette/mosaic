import rpyc

def download_file(url, path):
	con = rpyc.connect('localhost', port=1234)
	con._config['sync_request_timeout'] = None
	returned_path = con.root.download_file(url, path)
	return returned_path
