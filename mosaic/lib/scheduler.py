import re
import os
import json
import rpyc
import sys
import time
import subprocess

'''
    Role du scheduler:
        -> Lancer les pipes et récupérer les signaux
        -> Communiquer avec le serveur ServiceModule pour lui envoyer les informations:
            - de process_data
            - de status de processus
        -> Kill le server rpyc quand les runs sont finis
        -> Doit gérér la répartition des gpus
'''

class Scheduler():
    def __init__(self, pipelines, monitor_info, process_info=None, rerun_ids=None, rerun_params=None):
        self.monitor_info = monitor_info
        self.process_info = process_info
        self.pipelines = pipelines 
        self.rerun_ids = rerun_ids
        self.rerun_params = rerun_params
        self.procs = {}
        self.init_service_module()
        self.launch_pipelines()

    def init_service_module(self):
        self.smodule_proc = subprocess.Popen('serviceModule.py')

        while True:
            try:
                self.smodule_con = rpyc.connect('localhost', port=1234)
                self.smodule_con._config['sync_request_timeout'] = None
                break
            except: time.sleep(0.5)

        self.smodule_con.root.init_databases(self.monitor_info['database_path'], self.monitor_info['cache_database_path'])
        sizes = {'T':1e12, 'G':1e9, 'M':1e6, 'K':1e3}
        value = float(self.monitor_info['cache_size'][:-1])
        unit = sizes[self.monitor_info['cache_size'][-1]]
        cache_size = value * unit
        self.smodule_con.root.set_system_cache(cache_size)

    def launch_pipelines(self):
        self.nb_processus = int(self.monitor_info['nb_processus'])
        self.need_gpu = self.monitor_info['need_gpu'] == 'True'
        self.files_dir = self.process_info['run_files_path']
        if not self.files_dir.endswith('/'):
            self.files_dir += '/'
        if self.need_gpu:
            self.gpus_available = re.findall(r'[\w:]+', self.monitor_info['gpu_available'])
            self.gpus_dict = {gpu : None for gpu in self.gpus_available}
        i = 0
        paused = ''
        device = 'cpu'
        while i < len(self.pipelines) and not paused:
            if self.need_gpu:
                device = self._get_gpu()
            if len(self.procs) < self.nb_processus and (device != 'cpu' or not self.need_gpu):
                if self.rerun_ids is None:
                    self.run_id = self.smodule_con.root.get_next_id(self.monitor_info['database_path'])
                else:
                    self.run_id = self.rerun_ids[i]
                print(f' ---> launch {self.run_id} {i + 1}/{len(self.pipelines)}')
                if self.need_gpu:
                    self.gpus_dict[device] = self.run_id
                pipeline_serialized = json.dumps(self.pipelines[i])
    
                if self.rerun_ids is not None:
                    process_info_serialized = json.dumps(self.rerun_params[self.run_id])
                else:
                    process_info_serialized = json.dumps(self.process_info)
                
                if self.rerun_ids is None:
                    self.smodule_con.root.send_pipeline(self.run_id, pipeline_serialized)
                
                self.smodule_con.root.send_status(self.run_id, 'running', self.rerun_ids != None)
                self.smodule_con.root.update_runs_table(self.run_id, pipeline=self._format_pipeline(pipeline_serialized), status='running')
     
                os.makedirs(self.files_dir, exist_ok=True)
                log_file = open(f'{self.files_dir}output_{self.run_id}.log', 'w')
                p = subprocess.Popen(['execPipe.py', str(self.run_id), pipeline_serialized, process_info_serialized, device, str(int(self.rerun_ids != None))], stdout=log_file, stderr=log_file)
                self.procs[p] = {'log_file' : log_file, 'run_id' : self.run_id}
                i += 1
            self._check_procs_life()
            paused = self.smodule_con.root.is_paused()

        while len(self.procs) > 0:
            self._check_procs_life()

        if paused:
            print('paused', sys.argv[0])
            with open(paused, 'w') as f:
                json.dump([self.monitor_info,
                           self.rerun_params if self.rerun_ids is not None else self.process_info,
                           self.pipelines[i:],
                           self.rerun_ids is not None,
                           self.rerun_ids if self.rerun_ids is not None else None],
                           f, indent=4)

        try:
            self.smodule_con.root.exit()
        except:
            pass
    
    def _get_gpu(self):
        for gpu, proc_id in self.gpus_dict.items():
            if proc_id == None:
                return gpu
        return 'cpu'

    def _check_procs_life(self):
        new_procs = {}
        for proc in self.procs:
            return_status = proc.poll()
            _id = self.procs[proc]['run_id']
            if return_status is not None:
                if self.need_gpu:
                    for gpu, proc_id in self.gpus_dict.items():
                        if proc_id == _id:
                            self.gpus_dict[gpu] = None
                            break
                self.smodule_con.root.send_status(_id, 'end')
                self.smodule_con.root.update_runs_table(_id, status='end')
                self.procs[proc]['log_file'].close()
                print('[run finished]', _id)
            else:
                new_procs.update({proc:{'run_id' : _id, 'log_file' : self.procs[proc]['log_file']}})
        self.procs = new_procs

    def _format_pipeline(self, pipe):
        pipe = json.loads(pipe)
        res = []
        for elem in pipe:
            values = []
            for key, value in elem.items():
                if (key != 'name' and key != 'type' and key != 'path_to_class'
                and key != 'class' and key != 'key'):
                    values.append(str(value))
            res.append(elem['class'] + '(' + ','.join(values) + ')')
        return ' | '.join(res)

    def __del__(self):
        try:
            self.smodule_con.root.exit()
        except:
            pass
