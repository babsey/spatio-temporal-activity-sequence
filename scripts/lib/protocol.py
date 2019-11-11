# -*- coding: utf-8 -*-
#
# protocol.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import subprocess
import datetime
import json
import time

from sumatra import projects
import sumatra as smt

__all__ = [
    'get_parameters',
    'get_or_simulate',
]


def get_output_data(record):
    data = []

    for output_file in record.output_data:
        if output_file.metadata['mimetype'] != 'text/plain':
            continue
        d = np.loadtxt('./Data/' + output_file.path)
        if (d.ndim == 1) and (len(d) > 0):
            d = np.array([d])
        data.append(d)

    if len(data) > 0:
        data = np.concatenate(data)
        data = data[np.argsort(data[:, 1])]
        return data.T
    else:
        return [],[]



def run_bash_command(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def get_parameters(simulation, pardir='params'):
    params_filename = pardir + '/' + simulation + '.json'
    if params_filename:
        try:
            params = smt.parameters.build_parameters(params_filename)
        except:
            print('Parameter file not found.')
            params = smt.parameters.JSONParameterSet('{}')
    else:
        params = smt.parameters.JSONParameterSet('{}')
    return params


def run_simulation(main_file, parameters):
    print('Record does not exist. Create a new one ...')

    now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    new_params_filename = '%s.json' % now
    with open(new_params_filename, 'w') as f:
        f.write(json.dumps(parameters.as_dict()))

    run_bash_command('smt run -m %s %s' % (main_file, new_params_filename))
    time.sleep(1)
    run_bash_command('rm %s' % (new_params_filename))


def get_or_simulate(simulation, params_dict={}, pardir='simulation'):
    time.sleep(1)
    project = projects.load_project()

    parameters = get_parameters(simulation)
    parameters.update(params_dict)

    main_file = pardir + '/' + simulation + '.py'
    records = project.find_records(main_file=main_file, parameters=parameters)

    if len(records) == 0:
        run_simulation(main_file, parameters)
        data = get_or_simulate(simulation, parameters.as_dict())

    else:
        print('Get data from %s' % records[0].label)

        try:
            data = get_output_data(records[0])
        except:
            print('Data not found.')
            run_simulation(main_file, parameters)
            data = get_or_simulate(simulation, parameters.as_dict())

    return np.array(data)
