# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:28:33 2015

@author: Alberto
"""

import os
import sys
import numpy as np
from collections import OrderedDict

# TODO:
# - Support "file" types parameters... How?
# - Check value against provided lo and hi
# - Handle the cases where one or both lo/hi are not provided
# - Implement proactive behavior on visibility (eg, query if 'a' not on cmd line)
# - Support for list/array parameters!!
# - Overwrite parfile upon closing??


def params2header(params, header):
    """
    Write in the HISTORY commentary keyword the list of parameters
    used in the module
    """
    for k in params.keys():
        header['history'] = "{}={}".format(k, params[k])

class Parameter(object):
    def __init__(self, name, t, vis, val, lo, hi, com):

        self.name = name

        if vis == 'a':
            self.visibility = 'Ask'
        elif vis == 'h':
            self.visibility = 'Hidden'
        else:
            raise ValueError('Unknown visibility "{v}" specified for parameter {n}'.format(v=vis, n=self.name))

        if t == 'i':
            self.type = 'Integer'
        elif t == 'r':
            self.type = 'Real number'
        elif t == 's':
            self.type = 'String'
        elif t == 'b':
            self.type = 'Boolean'
            val = str(val)
        else:
            raise ValueError('Unknown type "{typ}" specified for parameter {n}'.format(typ=t, n=self.name))

        self.set_value(val)

        self.comment = com

    def set_value(self, val):

        if self.type == 'Integer':
            self.value = np.int(val)
        elif self.type == 'Real number':
            self.value = np.float64(val)
        elif self.type == 'String':
            self.value = str(val)
        elif self.type == 'Boolean':
            val = str(val)
            if val == '1' or val.lower() == 'true' or val.lower() == 'yes' or val.lower() == 'y':
                self.value = True
            if val == '0' or val.lower() == 'false' or val.lower() == 'no' or val.lower() == 'n':
                self.value = False


class ParameterSystem(object):
    def __init__(self, arglist, pfile_name=None):

        # Open parfile - store object should we wish to modify it in the future
        # If so, remember to change the file mode!!
        if pfile_name is None:
            # If a specific parfile is not supplied, look for a file with the name
            # of the invoked script and a '.par' extension
            script_name = os.path.basename(arglist[0])
            if str.endswith(script_name, '.py'):
                script_name = script_name[:-3]
            pfile_name = script_name + '.par'

            # Look for parfile in the current directory
            if pfile_name in os.listdir('.'):
                print('Trying to open parameter file:', os.path.abspath('/'.join([os.getcwd(), pfile_name])))
                try:
                    self._pfile = open(pfile_name, 'r')
                except IOError as e:
                    print("\nI/O error opening parameter file %s in folder %s.\n[Errno %i] %s\n" % (
                    e.filename, os.getcwd(), e.errno, e.strerror))
                    sys.exit()

            # Look for parfile indirectories specified by PFILES
            else:
                print('No parfile found in current directory. Looking into PFILES environment variable.')
                pfile_env = os.getenv('PFILES')
                if pfile_env is None:
                    print("No PFILES environment variable set")
                    print("No parfile could be found for " + script_name, file=sys.stderr)
                    sys.exit()
                else:
                    pfiledirs = pfile_env.replace(';', ':').split(":")
                    for d in pfiledirs:
                        if pfile_name in os.listdir(d):
                            print("Found %s in %s" % (pfile_name, d))
                            pfile_path = os.path.abspath('/'.join([d, pfile_name]))
                            print('Trying to open parameter file: ', pfile_path)
                            try:
                                self._pfile = open(pfile_path, 'r')
                            except IOError as e:
                                print("\nI/O error opening parameter file %s in folder %s.\n[Errno %i] %s\n" % (
                                e.filename, os.getcwd(), e.errno, e.strerror))
                                sys.exit()
                            break
                    else:
                        print("No parfile could be found for " + script_name, file=sys.stderr)
                        sys.exit()

        else:
            print('Trying to open parameter file: ', pfile_name)
            try:
                self._pfile = open(pfile_name, 'r')
            except IOError as e:
                print("\nI/O error opening parameter file %s in folder %s.\n[Errno %i] %s\n" % (
                e.filename, os.getcwd(), e.errno, e.strerror))
                sys.exit()

        # Read file lines
        plines = self._pfile.readlines()

        # Remove empty or comment lines
        plines = [l for l in plines if l != '\n']
        plines = [l for l in plines if l[0] != '#']

        # Split parameters properties
        plines = [l.split(',') for l in plines]

        # Initialize and populate parameter list
        self._params = OrderedDict()

        for par in plines:
            name = par[0].strip(' "')
            t = par[1].strip(' "')
            vis = par[2].strip(' "')
            val = par[3].strip(' "')
            lo = par[4].strip(' "')
            hi = par[5].strip(' "')
            com = par[6].strip(' "')

            self._params[name] = Parameter(name, t, vis, val, lo, hi, com)

        # Parse command line arguments for non-default parameter values
        for par in arglist[1:]:
            name, val = par.split('=', maxsplit=1)
            self._params[name].set_value(val)

    def __getitem__(self, key):

        return self._params[key].value

    def parameter(self, key):

        return self._params[key]

    def items(self):
        return self._params.items()

    def keys(self):
        return self._params.keys()

