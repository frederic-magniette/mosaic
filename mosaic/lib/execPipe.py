#!/usr/bin/env python3
# coding: utf-8

import re
import os
import sys
import rpyc
import time
import json
import torch
import inspect
import importlib
import traceback
import numpy as np
from tqdm import tqdm

class ExecPipe():
    def __init__(self, pipeline, info, rerun=False):
        self.pipeline = pipeline
        self.info = info
        self.rerun = rerun
        self.run_dir = self.info['run_files_path']
        if not self.run_dir.endswith('/'):
            self.run_dir += '/'
        self.database_path = self.info['database_path']
        self.run_info = {}
        self._classify_schemes()
        self._import()

    def _find_data_scheme(self):
        return re.split(r' ?, ?', self.info['data_scheme'])

    def _find_pipeline_scheme(self):
        return re.split(r' ?, ?', self.info['pipeline_scheme'])

    def _import(self):
        self.modules_pointers = {}
        for module in self.pipeline:
            path = module['path_to_class']
            sys.path.append(os.path.abspath(path))
            path = path.split('/')
            file = path[-1]
            path = path[:-1]
            sys.path.append('/'.join(path))
            mod = importlib.import_module(file.split('.')[0])
            self.modules_pointers[module['class']] = getattr(mod, module['class'])

    def _classify_schemes(self):
        self.classes = {'data':[], 'pipeline':[]}
        data_scheme = self._find_data_scheme()
        pipeline_scheme = self._find_pipeline_scheme()

        for module in self.pipeline:
            if module['type'] in data_scheme:
                self.classes['data'].append(module)
            elif module['type'] in pipeline_scheme:
                self.classes['pipeline'].append(module)

    def _get_params(self, module_class, run_info):
        args = inspect.signature(module_class)
        function_params = [param for param in args.parameters.values()]
        params = {}
        for param in function_params:
            if param.name in run_info:
                params[param.name] = run_info[param.name]
            else:
                i = 0
                found = 0
                while i < len(self.pipeline) and found == 0:
                    if param.name in self.pipeline[i]:
                        params[param.name] = self.pipeline[i][param.name]
                        found = 1
                    i += 1
                if param.default == param.empty and param.kind == param.POSITIONAL_OR_KEYWORD and found == 0:
                    raise ValueError(f'{param.name} with kind [{param.kind}] not found for {module_class.__name__}.')
        return params

    def _save_pipeline_info(self, run_id):
        root = f"./{self.run_dir}"
        with open(F'{root}run_info_{str(run_id)}.mdt', 'a') as f:
            f.write(f"pipeline={json.dumps(self.pipeline)}\n")
            f.flush()

    def _train(self, train_loader, crit, optimizer, device):
        train_loss, train_correct = 0.0, 0

        for module in self.pipeline_modules:
            module.train()

        for x, y in tqdm(train_loader, desc='train'):
            optimizer.zero_grad()

            for i in range(len(self.pipeline_modules)):
                module = self.pipeline_modules[i]
                module_info = self.classes['pipeline'][i]
                x = module.forward(x, self.run_info, module_info)

            y_pred = x.squeeze()
            y = y.squeeze()
            loss = crit(y, y_pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            train_correct += ((y.cpu().detach() > 0) == (y_pred.cpu().detach() > 0)).sum().item()
        
        return train_loss, train_correct

    def _test(self, test_loader, crit, device):
        test_loss, test_correct = 0.0, 0
        for module in self.pipeline_modules:
            module.train(False)

        for x, y in tqdm(test_loader, desc='test'):
            for i in range(len(self.pipeline_modules)):
                module = self.pipeline_modules[i]
                module_info = self.classes['pipeline'][i]
                x = module.forward(x, self.run_info, module_info)

            y_pred = x.squeeze()
            y = y.squeeze()
            loss = crit(y, y_pred)
            test_loss += loss.detach().item()
            test_correct += ((y.cpu().detach() > 0) == (y_pred.cpu().detach() > 0)).sum().item()

        return test_loss, test_correct

    def _prepare_data_scheme(self):
        data = None
        for module in self.classes['data']:
            module_class = self.modules_pointers[module['class']]
            params = self._get_params(module_class, self.run_info)
            module_class = module_class(**params)
            data = module_class.prepare(data, self.run_info, module)
            self.run_info.update(module_class.info())
        
        train_loader, test_loader = data
        self.run_info['train_loader'] = train_loader
        self.run_info['test_loader'] = test_loader

    def _prepare_pipeline_scheme(self):
        self.pipeline_modules = []
        for module in self.classes['pipeline']:
            module_class = self.modules_pointers[module['class']]
            params = self._get_params(module_class, self.run_info)
            module_class = module_class(**params)
            if self.rerun:
                module_class.load_model(f'{self.run_dir}model_{module_class.__class__.__name__}_{str(self.run_info["run_id"])}')
            self.pipeline_modules.append(module_class)
            self.run_info.update(module_class.info())

    def run(self, run_id, device):
        run_duration = time.perf_counter()
        self.run_info['run_id'] = int(run_id)
        self.run_info['device'] = device
        if not self.rerun:
            self._save_pipeline_info(run_id)

        self._prepare_data_scheme()
        self._prepare_pipeline_scheme()

        for modules in self.pipeline_modules:
            modules.to(self.run_info['device'])

        lr = float(self.info['lr'])
        optimizer = torch.optim.Adam(self.pipeline_modules[-1].parameters(), lr=lr)
        if 'loss_function' in self.info:
            crit = eval('torch.nn.' + self.info['loss_function'] + '().to(self.run_info["device"])')
        elif 'class_loss_function' in self.info and 'path_to_class_loss_function' in self.info:
            path = self.info['path_to_class_loss_function']
            sys.path.append(os.path.abspath(path))
            path = path.split('/')
            file = path[-1]
            path = path[:-1]
            sys.path.append('/'.join(path))
            mod = importlib.import_module(file.split('.')[0])
            crit = getattr(mod, self.info['class_loss_function'])
        else:
            raise Exception('No loss function provided in configuration file.')

        if self.rerun:
            history = torch.load(f'./{self.run_dir}history_{str(run_id)}.pt')
        else:
            history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

        epochs = int(self.info['epochs'])
        train_loss, train_acc, test_loss, test_acc = None, None, None, None
        for epoch in range(epochs):
            calc_duration_start = time.perf_counter()
            train_loss, train_correct = self._train(self.run_info['train_loader'], crit, optimizer, self.run_info['device'])
            test_loss, test_correct = self._test(self.run_info['test_loader'], crit, self.run_info['device'])
    
            len_train_loader = len(self.run_info['train_loader'])
            len_test_loader = len(self.run_info['test_loader'])

            batch_size = int(self.run_info['batch_size'])
            train_loss = train_loss / len_train_loader
            train_acc = train_correct / (len_train_loader * batch_size) * 100
            test_loss = test_loss / len_test_loader
            test_acc = test_correct / (len_test_loader * batch_size) * 100

            
            calc_duration = time.perf_counter() - calc_duration_start
            print(f"\nEPOCH {epoch:2} | train_loss {train_loss:.20f} | train_acc {train_acc:.2f}% | test_loss {test_loss:.20f} | test_acc {test_acc:.2f}% | {calc_duration:.4f}s\n")                
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            
            torch.save(history, f'./{self.run_dir}history_{str(run_id)}.pt')

            
            for model in self.pipeline_modules:
                model.save_model(f'{self.run_dir}/model_{model.__class__.__name__}_{str(run_id)}')

        run_duration = time.perf_counter() - run_duration

        with open(f'./{self.run_dir}run_info_{str(run_id)}.mdt', 'a') as f:
            f.write(f'\ntrain_loss={train_loss}')
            f.write(f'\ntrain_acc={train_acc}')
            f.write(f'\ntest_loss={test_loss}')
            f.write(f'\ntest_acc={test_acc}')
            f.flush()

        nb_params = [sum(param.numel() for param in model.parameters()) for model in self.pipeline_modules]

        data_to_save = self.info

        results = {'run_id'        : run_id,
                   'train_loss'    : train_loss,
                   'test_loss'     : test_loss,
                   'train_acc'     : train_acc,
                   'test_acc'      : test_acc,
                   'nb_params'     : sum(nb_params),
                   'duration(s)'   : run_duration,
                   'epochs': epochs}
        results.update(self._calc_learning_params(history))

        data_to_save['traceback'] = ''

        self._save_results(run_id, data_to_save, results, f'train_loss={train_loss:.3E}\ttest_loss={test_loss:.3E}\texecution_time={run_duration:.3E}')

    def _calc_learning_params(self, history):
        hist_test = np.array(history['test_loss'])
        hist_train = np.array(history['train_loss'])
        nb_epochs = len(hist_train)
        
        ten_p = nb_epochs // 10
        last_train = hist_train[nb_epochs - ten_p:]
        last_train_dec = hist_train[nb_epochs - ten_p - 1:-1]

        max_test = np.max(hist_test)
        max_train = np.max(hist_train)
        min_test = np.min(hist_test)
        min_train = np.min(hist_train)
        
        diff_hist = np.abs(hist_test - hist_train)
        max_data = np.max(np.array([max_test, max_train]))
        min_data = np.min(np.array([min_test, min_train]))
        range_data = max_data - min_data

        overfit = diff_hist[-1] / (range_data)
        trainability = np.sum(hist_test) / nb_epochs
        slope = np.mean(last_train - last_train_dec)
        return {'overfit': overfit, 'trainability': trainability, 'slope': slope}

    def _save_results(self, run_id, data_to_save, results, return_status):
        try:
            smodule_con = rpyc.connect('localhost', port=1234)
            smodule_con._config['sync_request_timeout'] = None
        except:
            print('failed at connection.')
            return
        smodule_con.root.send_info(run_id, json.dumps(data_to_save), self.rerun)
        smodule_con.root.send_results(run_id, json.dumps(results), self.rerun)
        smodule_con.root.update_runs_table(run_id, return_status=return_status)
        smodule_con.close()

if __name__ == '__main__':
    run_id = int(sys.argv[1])
    pipeline = json.loads(sys.argv[2])
    process_info = json.loads(sys.argv[3])
    device = sys.argv[4]
    rerun = int(sys.argv[5])
    try:
        pipe = ExecPipe(pipeline, process_info, rerun)
        pipe.run(run_id, device)
        smodule_con = rpyc.connect('localhost', port=1234)
        smodule_con._config['sync_request_timeout'] = None
        smodule_con.root.send_status(run_id, 'done')
        smodule_con.root.update_runs_table(run_id, status='done')
        smodule_con.close()
    except Exception as e:
        print(e)
        smodule_con = rpyc.connect('localhost', port=1234)
        smodule_con._config['sync_request_timeout'] = None 
        trace = traceback.format_exc()
        print(trace)
        smodule_con.root.send_status(run_id, 'error')
        smodule_con.root.update_runs_table(run_id, status='error', return_status=trace)
        smodule_con.root.send_traceback(run_id, trace)
        smodule_con.close()
