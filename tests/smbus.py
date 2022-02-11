#!/usr/bin/env python3
"""Mock smbus module

   Requirements
   python (3.5)
"""

import sys
import os
import json
import array

__version__ = '0.0'
__author__ = 'Fred Fierling'
__copyright__ = 'Copyright 2019, Spukhafte Systems Limited'

# Constants ====================

# Functions ====================

# Classes ======================
class Spec:
    def __init__(self, a, o, s):
        self.reg = a    # byte offset of register
        self.offset = o  # bit offset of field within register
        self.size = s    # size of field in bits

    def __str__(self):
        return '(%s, %s, %s)' % (self.reg, self.offset, self.size)

    __repr__ = __str__

class CDCE9xx():
    FNAME_FORMAT = '.cdce9%d%d_0.json'  # TODO separate state file for each address?

    BYTE_XFR = 0x80
    BASE_ADDR = 0x64

    EEPIP   = Spec(0x1, 6, 1)
    EELOCK  = Spec(0x1, 5, 1)
    SLAVE_ADDR = Spec(0x1, 0, 2)

    EEWRITE = Spec(0x6, 0, 1)

    EEPIP_COUNT = 3  # Number of times to read EEPIP == 1

    class state:
        IDLE = 0
        WRITE = 1

    def def_addr(self, n_plls):
        return self.BASE_ADDR + (((n_plls - 1) & 2) << 2) + (n_plls & 1)

    def __init__(self, fin, n_plls, fname=None):
        self.n_plls = n_plls
        self.state = CDCE9xx.state.IDLE
        self.fin = fin
        self.eepip = 0
        self.reg = array.array('B')
        self.base_addr = self.BASE_ADDR + (((n_plls - 1) & 2) << 2) + (n_plls & 1)

        if fname is None:
            self.fname = self.FNAME_FORMAT % (n_plls, ((n_plls * 2) + 1))
        else:
            self.fname = fname

        #if os.access(self.fname, os.R_OK):
        #    self.reg = array.array('B')
        try:
            with open(self.fname, 'r') as reg_file:
                self.reg.fromlist(json.load(reg_file))

        except (IOError, json.JSONDecodeError):
            # TI defaults
            self.reg.fromlist(
                    [1, 0, 0xB4, 1,    2, 0x50, 0xA0, 0, 0,  0, 0, 0, 0,  0, 0, 0] +
                    [0, 0,   0,  0, 0xED,    2,    1, 1, 0, 64, 2, 8, 0, 64, 2, 8] * n_plls
            )

        mask = (1 << CDCE9xx.SLAVE_ADDR.size) - 1
        # CDCE9xx.SLAVE_ADDR.offset == 0
        self.addr = self.base_addr | (self.reg[CDCE9xx.SLAVE_ADDR.reg] & mask)

    def __str__(self):
        return str(self.reg)

    def read(self, cmd):
        if cmd & CDCE9xx.BYTE_XFR:
            cmd ^= CDCE9xx.BYTE_XFR

            # Emulate special bits
            if cmd == CDCE9xx.EEPIP.reg:
                if self.eepip:
                    self.eepip -= 1
                    return self.reg[cmd] | (1 << CDCE9xx.EEPIP.offset)
            
            return self.reg[cmd]

        # Block transfers not emulated
        return None

    def write(self, cmd, val):
        if cmd & CDCE9xx.BYTE_XFR:
            cmd ^= CDCE9xx.BYTE_XFR

            # Check for read-only fields and special bits?
            if cmd == CDCE9xx.EEWRITE.reg:
                mask = 1 << CDCE9xx.EEWRITE.offset

                if mask & val and not mask & self.reg[cmd]:
                    # Positive edge detected on EEWRITE
                    self.eepip = self.EEPIP_COUNT

                    with open(self.fname, 'w') as reg_file:
                        json.dump(self.reg.tolist(), reg_file)

            self.reg[cmd] = val

            if cmd == CDCE9xx.SLAVE_ADDR.reg:
                mask = 1 << CDCE9xx.SLAVE_ADDR.offset
                self.addr = self.base_addr | (self.reg[CDCE9xx.SLAVE_ADDR.reg] & mask)
            
            return True

        # Block transfers not emulated
        return None

class SMBus:

    # Instantiate bus devices
    DEVICES = (
        CDCE9xx(30.72e6, 1),  # K15
        CDCE9xx(27.00e6, 2),  # K16r0
        CDCE9xx(27.00e6, 3),  # K16r1
        )

    def __init__(self, bus):
        '''Instantiate a mock SMBus object with a k15 and a k16'''

        self.opened = False
        self.bus = bus
        self.devices = {}
       
        for i in self.DEVICES:
            self.devices.update({i.addr: i})

    def open(self, bus):
        '''Connect object to specified bus'''

        if self.bus == bus:
            self.opened = True
            return None
        else:
            raise OverflowError

    def read_byte_data(self, addr, cmd):
        if self.opened:
            if addr in self.devices:
                return self.devices[addr].read(cmd)
            else:
                raise IOError
        else:
            raise IOError

    def write_byte_data(self, addr, cmd, val):
        if self.opened:
            if addr in self.devices:
                device = self.devices[addr]
                rval = device.write(cmd, val)

                if device.addr != addr:
                    # Device's address changed, change its key too
                    self.devices[device.addr] = self.devices.pop[addr]

                return rval

            else:
                raise IOError
        else:
            raise IOError

    def close(self):
        self.opened = False
        return True
