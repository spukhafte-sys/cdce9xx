#!/usr/bin/env python3
"""Identify hardware platform
"""

__version__ = '0.0'
__author__ = 'Fred Fierling'
__copyright__ = 'Copyright 2019, Spukhafte Systems Limited'

import io
import sys
import math
from itertools import compress

# Constants
CPU_INFO = '/proc/cpuinfo'

HARDWARE = {
    'pi':    ('BCM2708',
              'BCM2709',
              'BCM2835',
              'BCM2836',
             ),

    'bb-b':  ('Generic AM33XX',),
    'bb-ai': ('Generic DRA74X',),
}

# ==================================
def primes(n):
    """Return a list of primes < n for n > 2."""
    sieve = bytearray([True]) * (n//2)

    for i in range(3, int(n**0.5)+1, 2):
        if sieve[i//2]:
            sieve[i*i//2::i] = bytearray((n-i*i-1)//(2*i)+1)

    return [2, *compress(range(3, n, 2), sieve[1:])]

def factorize(n):
    """Return list of n's prime factors and their exponents."""
    pf = []

    for p in primes(int(n**0.5)+1):
        if p*p > n:
            break
        count = 0
        while not n%p:
            n //= p
            count += 1
        if count > 0:
            pf.append((p, count))

    if n > 1:
        pf.append((n, 1))

    return pf

def guess_platform():
    """Guess hardware platform"""

    try:
        with io.open(CPU_INFO, 'r') as cpuinfo:
            hw_valid = False
            for line in cpuinfo:
                if line.startswith('Hardware'):
                    hw_valid = True
                    _, hw_info = line.strip().split(':', 1)
                    hw_info = hw_info.strip()

                    for platform in HARDWARE:
                        for i in HARDWARE[platform]:
                            if i in hw_info:
                                return(platform)

            if not hw_valid:
                return False

    except IOError:
        return False
