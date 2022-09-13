#!/usr/bin/env python3
#
# Unit tests for cdce9xx.py

__version__ = '0.9.1'
__author__ = 'Fred Fierling'
__copyright__ = 'Copyright 2022, Spukhafte Systems Limited'

TEST_DATA_FILENAME = 'tests/data/cdce9xx.csv'
NUM_TEST_VECTORS = 1024

import unittest
import csv

from random import randrange
from math import ceil

from spukhafte.cdce9xx import CDCE9xx, PLL
from tests.smbus import SMBus

class Test_find_n_m_pdiv(unittest.TestCase):
    @unittest.skip("Only use this test to generate test data")
    def test_gen_data_find_n_m_pdiv(self):
        f = open(TEST_DATA_FILENAME, 'w')

        with f:
            data = csv.writer(f)

            # Sanity checks
            for fin, fout in (
                    (30.72e6, 10e6),
                    (30.72e6, 20e6),
                    (30.72e6, 122858447),
                    (30.72e6, 107540849),
                    (27.0e6, 10e6),
                    (27.0e6, 20e6),
                    ):

                data.writerow((fin, fout) + PLL.find_n_m_pdiv(fin, fout, PLL.MAX_PDIV1))

            i = NUM_TEST_VECTORS
            while i:
                i -= 1
                fin = randrange(CDCE9xx.MIN_FIN, CDCE9xx.MAX_FIN)
                fout = randrange(ceil(PLL.MIN_VCO/PLL.MAX_PDIV), PLL.MAX_VCO)
                data.writerow((fin, fout) + PLL.find_n_m_pdiv(fin, fout, PLL.MAX_PDIV1))

            f.close()

    @unittest.skip("Only use this test to generate test data")
    def test_gen_data_factory_default(self):
        f = open(TEST_DATA_FILENAME, 'r')

        with f:
            data = csv.writer(f)

    def test_nm_0(self):
        self.assertEqual(PLL.find_n_m_pdiv(30.72e6, 10e6, PLL.MAX_PDIV1),
                        (2875, 384, 23, 230000000, 0))

    def test_find_n_m_pdiv_check_data(self):
        import pdb; pdb.set_trace
        f = open(TEST_DATA_FILENAME, 'r')

        with f:
            i = 0
            data = csv.reader(f)

            for fin, fout, n, m, pdiv, fvco, error in data:
                i += 1
                fin = float(fin)
                fout = float(fout)
                fvco = float(fvco)
                vector = (int(n), int(m), int(pdiv), float(fvco), float(error))

                self.assertEqual(PLL.find_n_m_pdiv(fin, fout, PLL.MAX_PDIV1), vector)

            print('\nProcessed %d vectors' % i)

            f.close()

