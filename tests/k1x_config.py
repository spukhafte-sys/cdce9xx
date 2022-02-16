#!/usr/bin/env python3
#
# Unit tests for k15_config and k16_config

__version__ = '0.8.0'
__author__ = 'Fred Fierling'
__copyright__ = 'Copyright 2019, Spukhafte Systems Limited'

import unittest
import json
import os
from subprocess import Popen, PIPE

from random import randrange
from math import ceil
#from datetime import datetime

TEST_DATA_FILENAME = 'tests/.k1x_config.json'
TEST_ENV = { 'PYTHONPATH': './tests' }  # Path that finds mock smbus first
K15_CMD = './tests/k15_config'
K16_CMD = './tests/k16_config'

STIMULI = [
          # [stdin, [cmd, *opts, ]] ; stdin is null JSON because test env doesn't look like a tty
            ['{}', [K15_CMD, '-h',]],  # Check help
            ['{}', [K15_CMD, '-r',]],  # Set to chip default
            ['{}', [K15_CMD, '5e6',]],
            ['{}', [K15_CMD, '-d2', '1e7',]],
            ['{}', [K15_CMD, '-jz',]],
            ['{}', [K15_CMD, '-s1',]],
            ['{}', [K15_CMD, '-p0',]],
            ['{}', [K15_CMD, '-v',]],
            ['{}', [K15_CMD, '-nmp', '2875', '384', '46',]],
            ['{"PDIV2": 46}', [K15_CMD, '-j',]],
            ['{}', [K15_CMD, '-vw',]],  # Write

            ['{}', [K16_CMD, '-h',]],
            ['{}', [K16_CMD, '-r',]],
            ['{}', [K16_CMD, '1e7',]],
            ['{}', [K16_CMD, '-jz',]],
            ['{"XCSEL": 5}', [K16_CMD, '-j',]],
            ['{}', [K16_CMD, '-vw',]],
          ]

class Test_k1x_config(unittest.TestCase):
    def setup(self):
        pass

    @unittest.skip("Run this test only to generate test data")
    def test_00_k1x_config_default(self):
       #import pdb; pdb.set_trace()
        with open(TEST_DATA_FILENAME, 'w') as f:
            out = []
            for i in STIMULI:
                p1 = Popen(i[1], env=TEST_ENV, stdin=PIPE, stdout=PIPE, stderr=PIPE)

                resp = p1.communicate(input=i[0].encode('ascii'))
                out.append(i + [" ".join(resp[j].decode('utf-8').split()) for j in (0,1)])

            json.dump(out, f, indent=1)

    def test_01_k1x_config(self):
        with open(TEST_DATA_FILENAME, 'r') as f:
           #import pdb; pdb.set_trace()
            for i in json.load(f):
                p1 = Popen(i[1], env=TEST_ENV, stdin=PIPE, stdout=PIPE, stderr=PIPE)

                resp = p1.communicate(input=i[0].encode('ascii'))

                self.assertEqual(i[2], " ".join(resp[0].decode('utf-8').split()))
                self.assertEqual(i[3], " ".join(resp[1].decode('utf-8').split()))
