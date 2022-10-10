#!/usr/bin/env python3
# coding: utf-8

from argparse import ArgumentParser
from configparser import ConfigParser
import re
import copy
import json

class PipelinesGenerator():

    def __init__(self, config_path, database_path=None):
        self.config_path = config_path
        self.database_path = database_path
        self.config = ConfigParser()
        self.config.read(self.config_path)

    def _match_to_ints(self, match):
        return map(int, match.group(1).split('-'))

    def __findall_intrange_combinations(self, sections, section, section_name):
        range_pattern = re.compile(r"\{([0-9]+\-[0-9]+)\}")
        range_found = 0
        for key, values in section.items():
            result = range_pattern.match(str(values))
            if result:
                range_found = 1
                _min, _max = self._match_to_ints(result)
                for elem in range(_min, _max + 1):
                    new_section = copy.deepcopy(dict(section))
                    if not 'name' in section:
                        new_section['name'] = section_name
                    new_section[key] = elem
                    self.__findall_intrange_combinations(sections, new_section, section_name)
        if range_found == 0:
            if section not in sections:
                sections.append(section)

    def __findall_list_combinations(self, sections, section, section_name):
        li_found = 0
        for key, values in section.items():
            li = re.findall(r'[\w\.\/\:{}-]+', str(values))
            if len(li) > 1:
                li_found = 1
                for elem in li:
                    new_section = copy.deepcopy(dict(section))
                    if not 'name' in section:
                        new_section['name'] = section_name
                    new_section[key] = elem
                    self.__findall_list_combinations(sections, new_section, section_name)
        if li_found == 0:
            if section not in sections:
                sections.append(section)

    def _check_ranges(self, config_section, section_name):
        sections = []
        new_sections = []
        self.__findall_intrange_combinations(new_sections,
                                                config_section,
                                                section_name)
        for new_conf in new_sections:
            self.__findall_list_combinations(sections,
                                             new_conf,
                                             section_name)
        return sections

    def _mix_modules(self, objs, types, modules, pipelines):
        if not types:
            pipelines.append(modules)
            return
        for module in objs[types[0]]:
            self._mix_modules(objs, types[1:], modules + [module], pipelines)

    def _find_scheme(self):
        scheme = ','.join([self.config['PROCESS']['data_scheme'], self.config['PROCESS']['pipeline_scheme']])
        return re.split(r' ?, ?', scheme)

    def _multiply_pipelines(self, pipelines):
        multiplied_pipelines = []
        multiplicity = int(self.config['MONITOR']['multiplicity'])
        for pipeline in pipelines:
            multiplied_pipelines.extend([pipeline for i in range(multiplicity)])
        return multiplied_pipelines
    
    def _check_key(self, pipelines):
        keys_pattern = r"{(\w+)}*"
        words_pattern = r"(?<=\})(.*?)(?=\{|$)"
        for pipe in pipelines:
            if 'key' in pipe:
                matches = re.finditer(keys_pattern, pipe['key'])
                words = re.findall('^\w*', pipe['key']) + re.findall(words_pattern, pipe['key'])
                sections = [{'key':match.group()[1:-1], 'pos':match.span()} for match in matches]
                section_begin = False
                if sections and sections[0]['pos'][0] == 0:
                    section_begin = True
                _all = []
                i = 0
                j = 0
                count = 0
                while i < len(sections) or j < len(words):
                        if count % 2 == (not section_begin) and i < len(sections):
                                _all.append(str(pipe[sections[i]['key']]))
                                i += 1
                        elif count % 2 == section_begin and j < len(words):
                                _all.append(str(words[j]))
                                j += 1
                        count += 1
                pipe['key'] = ''.join(_all)
        return pipelines


    def create_pipelines(self):
        objs = {}
        for section in self.config.sections():
            if 'type' in self.config[section].keys():
                sections_ranges = self._check_ranges(copy.deepcopy(dict(self.config[section])), section)
                sections_ranges = self._check_key(sections_ranges)
                if self.config[section]['type'] not in objs:
                    objs[self.config[section]['type']] = sections_ranges
                else:
                    objs[self.config[section]['type']].extend(sections_ranges)

        if not objs:
            return []
        pipelines = []
        self._mix_modules(objs, self._find_scheme(), [], pipelines)
        pipelines = self._multiply_pipelines(pipelines)
        return pipelines

    def get_monitor_info(self):
        if not 'MONITOR' in self.config.sections():
            raise ValueError(f'MONITOR section not found in {self.config_path}')
        elif not 'PROCESS' in self.config.sections():
            raise ValueError(f'PROCESS section not found in {self.config_path}')
        monitor_config = dict(self.config['MONITOR'])
        process_config = dict(self.config['PROCESS'])
        monitor_config['database_path'] = self.database_path
        process_config['database_path'] = self.database_path
        return monitor_config, process_config

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_file', type=str)
    parser.add_argument('dest_json_file', type=str)
    args = vars(parser.parse_args())

    if args["dest_json_file"].split('.')[-1] != 'json':
        print(f'{args["dest_json_file"]} is not a json file.')
        exit(1)

    pgen = PipelinesGenerator(args['config_file'])
    pipelines = pgen.create_pipelines()
    if pipelines:
        with open(args["dest_json_file"], 'w') as f:
            json.dump(pipelines, f, indent=4)
            print(f'Pipelines saved in {args["dest_json_file"]}.')
