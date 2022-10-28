#!/usr/bin/env python3
# coding: utf-8

import rpyc
import json
import sys
import os
import time
import sqlite3
from urllib import request
from rpyc.utils.server import ThreadedServer
from threading import Lock
from shutil import copy2

'''
    Role du ServiceModule:
        -> Faire le lien entre le scheduler / les processus / la db
        -> Gérer le cache (secondaire)
        -> Pouvoir se close quand le scheduler lui envoie l'ordre

TODO: Enlever le global_dic
      Enlever la fonction d'initialisation des informations du monitor + stocker le chemin de la db au moment de l'initialisation
'''

global_dic = {'process_data':{}, 'monitor_info':{}}
pause = ''

mutex = Lock()
mutex.acquire()

mutex_file = Lock()
mutex_file.acquire()
global_cache_path = ''
global_database_path = ''
queue = set()

class ServiceModule(rpyc.Service): 

    def exposed_save_path_file(self, path):
        global mutex_file, global_cache_path
        mutex_file.acquire()
        con = sqlite3.connect(global_cache_path)
        cur = con.cursor()
        file_size = os.stat(path).st_size
        cur.execute('''INSERT INTO cache VALUES (?, DATETIME("now"), ?, ?, ?)''', ('', file_size, path, 'no'))
        
        max_size, current_size = cur.execute('''SELECT max_size, current_size FROM system''').fetchone()
        new_size = current_size + file_size
        if current_size + file_size > max_size:
            diff_size = current_size + file_size - max_size
            paths, sizes = zip(*cur.execute('''
                SELECT path, size
                FROM   cache
                ORDER  BY last_use ASC
            ''').fetchall())

            sum_sizes = 0
            i = 0
            deleted_path = []
            while sum_sizes < diff_size:
                sum_sizes += sizes[i]
                deleted_path.append((paths[i],))
                os.remove(paths[i])
                i += 1
            cur.executemany('''DELETE FROM cache WHERE path = ?''', deleted_path)
            new_size = current_size + file_size - sum_sizes

        cur.execute('''UPDATE system SET current_size = ?''', (new_size,))
        con.commit()
        con.close()
        mutex_file.release()
    
    def exposed_get_path_file(self, path):
        global mutex_file, global_cache_path
        mutex_file.acquire()
        con = sqlite3.connect(global_cache_path)
        cur = con.cursor()
        db_path = cur.execute('SELECT path from cache WHERE path = ?', (path,)).fetchone()
        con.close()
        mutex_file.release()
        if db_path is None:
            return False
        return True

    def exposed_download_file(self, url, path):
        global mutex_file, global_cache_path
        while url in queue:
            time.sleep(0.1)
        mutex_file.acquire()
        con = sqlite3.connect(global_cache_path)
        cur = con.cursor()
        db_path = cur.execute('SELECT path FROM cache WHERE url = ?', (url, )).fetchone()
        if not db_path or not os.path.exists(db_path[0]):
            db_path = None
        con.close()
        mutex_file.release()
        if not db_path:
            queue.add(url)
            path_wo_file = '/'.join(path.split('/')[:-1])
            os.makedirs(path_wo_file, exist_ok=True)
            print('[+] Download : ', url)
            path = os.path.abspath(path)
            try:
                request.urlretrieve(url, path)
            except:
                print(f'URL {url} not found.')
                return None
            print('Download finished')
            file_size = os.stat(path).st_size

            mutex_file.acquire()
            con = sqlite3.connect(global_cache_path)
            cur = con.cursor()
            cur.execute('''INSERT INTO cache VALUES (?, DATETIME("now"), ?, ?, ?)''', (url, file_size, path, 'yes'))
            max_size, current_size = cur.execute('''SELECT max_size, current_size FROM system''').fetchone()

            new_size = current_size + file_size
            if current_size + file_size > max_size:
                diff_size = current_size + file_size - max_size
                paths, sizes = zip(*cur.execute('''
                    SELECT path, size
                    FROM   cache
                    ORDER  BY last_use ASC
                ''').fetchall())

                sum_sizes = 0
                i = 0
                deleted_path = []
                while sum_sizes < diff_size:
                    sum_sizes += sizes[i]
                    deleted_path.append((paths[i],))
                    os.remove(paths[i])
                    i += 1
                cur.executemany('''DELETE FROM cache WHERE path = ?''', deleted_path)
                new_size = current_size + file_size - sum_sizes

            db_path = path
            cur.execute('''UPDATE system SET current_size = ?''', (new_size,))
            con.commit()
            con.close()
            queue.remove(url)
            mutex_file.release()

        return db_path

    def exposed_get_next_id(self, database_path):
        global mutex
        mutex.acquire()
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        next_id = db_cur.execute('''SELECT param_value FROM system WHERE param_name="next_run_id"''').fetchone()[0]
        db_cur.execute('''UPDATE system SET param_value=? WHERE param_name="next_run_id"''', (next_id + 1, ))
        db_con.commit()
        db_con.close()
        mutex.release()
        return next_id

    def exposed_send_results(self, run_id, data, rerun=False):
        data = json.loads(data)
        self.save_results(run_id, data, rerun)
    
    def exposed_send_info(self, run_id, info, rerun=False):
        info = json.loads(info)
        self.save_info(run_id, info, rerun)

    def exposed_send_pipeline(self, run_id, pipeline):
        self.save_pipeline(run_id, pipeline)

    def exposed_send_status(self, run_id, status, rerun=False):
        global mutex, global_database_path
        mutex.acquire()
        database_path = global_database_path
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        old_status = db_cur.execute('''SELECT param_value FROM params WHERE (param_name="status" AND run_id=?)''', (run_id, )).fetchone()
        if status == 'running':
            if rerun:
                db_cur.execute('''UPDATE params SET param_value=? WHERE (run_id=? AND param_name="status")''', (status, run_id))
            else:
                db_cur.execute('''INSERT INTO params VALUES (?, ?, ?)''', (run_id, "status", status))
            db_con.commit()
        elif status == 'end' and old_status is not None and old_status[0] == 'running':
            db_cur.execute('''UPDATE params SET param_value=? WHERE (param_name="status" AND run_id=?)''', ('system failure', run_id))
            db_con.commit()
        elif status != 'end':
            db_cur.execute('''UPDATE params SET param_value=? WHERE (param_name="status" AND run_id=?)''', (status, run_id))
            db_con.commit()
        db_con.close()
        mutex.release()

    def exposed_send_traceback(self, run_id, traceback, rerun=False):
        global mutex, global_database_path
        mutex.acquire()
        database_path = global_database_path
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        if rerun:
            db_cur.execute('''UPDATE params SET param_value=? WHERE (run_id=? AND param_name="traceback")''', (traceback, run_id))
        else:
            db_cur.execute('''INSERT INTO params VALUES (?, ?, ?)''', (run_id, 'traceback', traceback))
        db_con.commit()
        db_con.close()
        mutex.release()

    def exposed_update_runs_table(self, run_id, status='undefined', pipeline='undefined', return_status='undefined'):
        global mutex, global_database_path
        mutex.acquire()
        database_path = global_database_path
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        _all = db_cur.execute('''SELECT * FROM runs WHERE run_id=?''', (run_id, )).fetchone()
        if _all is None:
            db_cur.execute('''INSERT INTO runs VALUES (?, ?, ?, ?)''', (run_id, status, pipeline, return_status))
            db_con.commit()
        else:
            if status != 'undefined':
                old_status = _all[1]
                if status == 'end' and old_status is not None and old_status[0] == 'running':
                    db_cur.execute('''UPDATE runs SET status=? WHERE run_id=?''', ('system_failure', run_id))
                    db_con.commit()
                elif status != 'end':
                    db_cur.execute('''UPDATE runs SET status=? WHERE run_id=?''', (status, run_id))
                    db_con.commit()
            if return_status != 'undefined':
                db_cur.execute('''UPDATE runs SET return_status=? WHERE run_id=?''', (return_status, run_id))
                db_con.commit()

        db_con.close()
        mutex.release()

    def save_info(self, run_id, data, rerun):
        global mutex, global_database_path
        mutex.acquire()
        database_path = global_database_path
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        for key, value in data.items():
            if key != 'run_id':
                if rerun:
                    db_cur.execute('''UPDATE params SET param_value=? WHERE (run_id=? AND param_name=?)''', (value, run_id, key))
                else:
                    db_cur.execute('''INSERT INTO params VALUES (?, ?, ?)''', (run_id, key, value))
        db_con.commit()
        db_con.close()
        mutex.release()

    def save_results(self, run_id, data, rerun):
        global mutex, global_database_path
        mutex.acquire()
        database_path = global_database_path
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        if rerun:
            old_epochs = int(db_cur.execute('''SELECT epochs FROM run_results WHERE run_id=?"''', (run_id, )).fetchone()[0])
            db_cur.execute('''UPDATE run_results SET run_id=?, train_loss=?, test_loss=?, train_acc=?, test_acc=?, nb_params=?, duration(s)=?, epochs=?,
                              overfit=?, trainability=?, slope=?''',
                            [run_id,
                            data['train_loss'],
                            data['test_loss'],
                            data['train_acc'],
                            data['test_acc'],
                            data['nb_params'],
                            data['duration(s)'],
                            int(data['epochs']) + old_epochs,
                            data['overfit'],
                            data['trainability'],
                            data['slope']])
        else:
            db_cur.execute('''INSERT INTO run_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                [run_id,
                                data['train_loss'],
                                data['test_loss'],
                                data['train_acc'],
                                data['test_acc'],
                                data['nb_params'],
                                data['duration(s)'],
                                data['epochs'],
                                data['overfit'],
                                data['trainability'],
                                data['slope']])
        db_con.commit()
        db_con.close()
        mutex.release()

    def save_pipeline(self, run_id, pipeline):
        global mutex, global_database_path
        mutex.acquire()
        database_path = global_database_path
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        db_cur.execute('''INSERT INTO params VALUES (?, ?, ?)''', (run_id, 'pipeline', pipeline))
        pipeline = json.loads(pipeline)
        for elem in pipeline:
            for key, value in elem.items():
                if key == 'key':
                    db_cur.execute('''INSERT INTO params VALUES (?, ?, ?)''', (run_id, elem['type'] + '_key', value))
                else:
                    db_cur.execute('''INSERT INTO params VALUES (?, ?, ?)''', (run_id, elem['class'] + '_' + key, value))
        db_con.commit()
        db_con.close()
        mutex.release() 

    def exposed_get_status_done(self):
        global global_database_path, mutex
        database_path = global_database_path
        mutex.acquire()
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        return_request = db_cur.execute('''SELECT run_id, return_status FROM runs WHERE status="done" ORDER BY run_id ASC''').fetchall()
        return_dic = {}
        for run_id, return_status in return_request:
            return_dic[run_id] = return_status
        db_con.close()
        mutex.release()
        return json.dumps(return_dic)

    def exposed_get_status_error(self):
        global global_database_path, mutex
        database_path = global_database_path
        mutex.acquire()
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        return_request = db_cur.execute('''SELECT run_id, pipeline, return_status FROM runs WHERE status="error" ORDER BY run_id ASC''').fetchall()
        return_dic = {}
        for run_id, pipeline, return_status in return_request:
            return_dic[run_id] = pipeline + '\n\t' + return_status.replace('\n', '\n\t')
        db_con.close()
        mutex.release()
        return json.dumps(return_dic)

    def exposed_get_status_running(self):
        global global_database_path, mutex
        database_path = global_database_path
        mutex.acquire()
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        return_request = db_cur.execute('''SELECT run_id, pipeline
                                           FROM runs WHERE status="running"
                                           ORDER BY run_id ASC''').fetchall()
        return_dic = {}
        for run_id, pipeline in return_request:
            return_dic[run_id] = pipeline
        db_con.close()
        mutex.release()
        return json.dumps(return_dic)

    def exposed_set_system_cache(self, max_size):
        global mutex_file, global_cache_path
        mutex_file.acquire()
        cache_con = sqlite3.connect(global_cache_path)
        cache_cur = cache_con.cursor()
        old_max_size = cache_cur.execute('''SELECT max_size FROM system''').fetchone()
        if old_max_size is None:
            cache_cur.execute('''INSERT INTO system VALUES (?, 0)''', (max_size, ))
        else:
            cache_cur.execute('''UPDATE system SET max_size = ?''', (max_size, ))
        cache_con.commit()
        mutex_file.release()

    def exposed_init_databases(self, database_path, cache_path):
        global mutex, global_cache_path, global_database_path
        global_cache_path = cache_path
        global_database_path = database_path
        db_con = sqlite3.connect(database_path)
        db_cur = db_con.cursor()
        try:
            next_run_id = int(db_cur.execute('''SELECT param_value
                                                FROM system
                                                WHERE param_name="next_run_id"''').fetchone()[0])
        except:
            db_cur.execute('''CREATE TABLE params (run_id, param_name, param_value)''')
            db_cur.execute('''CREATE TABLE IF NOT EXISTS run_results (run_id, train_loss, test_loss, train_acc, test_acc,
                              nb_params, "duration(s)", epochs, overfit, trainability, slope)''')
            db_cur.execute('''CREATE TABLE system (param_name, param_value)''')
            db_cur.execute('''CREATE TABLE runs (run_id, status, pipeline, return_status)''')
            db_cur.execute('''INSERT INTO system VALUES ("next_run_id", 1)''')
            db_con.commit()
        

        cache_con = sqlite3.connect(cache_path)
        cache_cur = cache_con.cursor()

        cache_cur.execute('''CREATE TABLE IF NOT EXISTS cache (url, last_use, size, path, downloading)''')
        cache_cur.execute('''CREATE TABLE IF NOT EXISTS system (max_size, current_size)''')
        cache_con.commit()

        db_con.close()
        cache_con.close()
        mutex.release()
        mutex_file.release()

    def exposed_set_paused(self, filename):
        global pause
        pause = filename

    def exposed_is_paused(self):
        global pause
        if pause:
            ret_pause = pause
            pause = ''
            return ret_pause
        return ''

    def exposed_save_database(self, path):
        global mutex, global_database_path
        database_path = global_database_path
        mutex.acquire()
        if path != database_path:
            copy2(database_path, path)
            mutex.release()
            return True
        mutex.release()
        return False

    def exposed_exit(self):
        global threaded_server, mutex
        mutex.acquire()
        threaded_server.close()

if __name__ == '__main__':
    try: 
        if len(sys.argv) > 1:
            global_database_path = sys.argv[1]
        threaded_server = ThreadedServer(ServiceModule, 'localhost', port=1234)
        threaded_server.start()
    except Exception as e:
        print(e)
