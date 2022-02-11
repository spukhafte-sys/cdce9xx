#!/usr/bin/env python3
"""Classes for Texas Instruments CDCE9xx clock generators and CDC3RL02 clock buffers.
"""

__version__ = '0.0'
__author__ = 'Fred Fierling'
__copyright__ = 'Copyright 2019, Spukhafte Systems Limited'

import sys
import json
import math
from collections import OrderedDict

# Constants ===================

# Functions ===================
def harmonic_in_band(fvco, bands):
    '''Returns description of band if fvco is within band, otherwise None.'''

    for lower, upper, desc in bands:
        if int(lower/fvco) != int(upper/fvco):
            return desc
    return None

def print_json(names_values):
    '''Prints names_values as JSON.
    :param names_values: A dictionary of names: values
    '''

    print('{')

    i = len(names_values)
    for name, value in names_values:
        i -= 1
        eol = ',' if i else ''
        print('  %-14s:%4d%s' % (('"%s"' % name), value, eol))

    print('}')
    return

# Classes ======================
class Spec:
    '''Class for bit field specification.

    :param reg: byte offset of register within device
    :param offset: bit offset of field within register
    :param size: size of field in bits
    '''

    def __init__(self, a, o, s):
        self.reg = a    # byte offset of register
        self.offset = o  # bit offset of field within register
        self.size = s    # size of field in bits

    def __str__(self):
        return '(%s, %s, %s)' % (self.reg, self.offset, self.size)

    __repr__ = __str__

class Field:
    '''Class for bit fields.

    :param r_seq: priority of field during batch reads (0: read-protected; 1: highest)
    :param w_seq: priority of field during batch writes (0: write-protected; 1: highest)
    :param d_val: device default value
    :param specs: field specifications
    '''

    def __init__(self, r, w, d, s):
        self.r_seq = r  # read sequence; 1:first; 0:read-protected
        self.w_seq = w  # write sequence; 1:first; 0:write-protected
        self.d_val = d  # default value
        self.specs = s  # list of field specifications

    def __str__(self):
        return '(%s, %s, %s, %s)' % (self.r_seq, self.w_seq, self.d_val, self.specs)

    __repr__ = __str__

class CDC3RL02:
    '''Class for a CDC3RL02 clock buffer.'''

    # CDC3RL02 Specifications
    MIN_CLK = 10e6
    MAX_CLK = 52e6

class PLL:
    '''Class for a CDCE9xx PLL.'''

    # PLL Constants
    VCO_RANGE = (
            (80e6, 125e6),
            (125e6, 150e6),
            (150e6, 175e6),
            (175e6, 230e6)
        )
    
    MIN_VCO = VCO_RANGE[0][0]
    MAX_VCO = VCO_RANGE[3][1]
    
    MIN_M = 1
    MAX_M = 511
    MIN_N = 1
    MAX_N = 4095
    
    # Register field limits
    MIN_PDIV = 1
    MAX_PDIV = 127
    MAX_PDIV1 = 1023
    MIN_P = 0
    MAX_P = 7
    MIN_Q = 16
    MAX_Q = 63
    MIN_R = 0
    MAX_R = 511
    
    FIELD_MIN = {
            'PDIV1': 1,
            'PDIV2': 1,
            'PDIV3': 1,
            'Q': 16,
            }

    @classmethod
    def fout_valid(cls, f, pdiv10b=False):
        '''Return true if f is valid fout.
        :param f: Frequency in Hz
        :param pdiv10b True if PLL divider is 10-bits
        '''
    
        if pdiv10b:
            pdiv = cls.MAX_PDIV1
        else:
            pdiv = cls.MAX_PDIV
    
        if cls.MIN_VCO/pdiv <= f <= cls.MAX_VCO:
            return True
    
        return False

    @classmethod
    def nmp_valid(cls, nmp, pdiv10):
        '''Returns true for a valid combination of N, M, and P values in the list nmp.

        :param nmp: list of (N, M, P)
        :param pdiv10: boolean true for 10-bit PDIV
        '''

        n, m, p = nmp
        max_pdiv = cls.MAX_PDIV1 if pdiv10 else cls.MAX_PDIV
    
        return (
            (isinstance(n, int) and cls.MIN_N <= n <= cls.MAX_N) and
            (isinstance(m, int) and cls.MIN_M <= m <= cls.MAX_M) and
            (isinstance(p, int) and cls.MIN_PDIV <= p <= max_pdiv)
        )
    
    @classmethod
    def pdivs(cls, f, max_pdiv=MAX_PDIV):
        """Yield the pdivs, in descending order, that could generate frequency f
        from the VCO's frequency range.
        :param f: frequency in Hz
        :param max_pdiv:
        """
    
        i = min(max_pdiv, math.floor(cls.VCO_RANGE[3][1]/f))
    
        while i >= max(cls.MIN_PDIV, math.ceil(cls.VCO_RANGE[0][0]/f)):
            yield i
            i -= 1
    
    @classmethod
    def vco_range(cls, f):
        """Return index of highest VCO_RANGE that includes frequency f."""

        i = 4
        while i:
            i -= 1
            if cls.VCO_RANGE[i][0] <= f <= cls.VCO_RANGE[i][1]:
                return i
    
        # f out of range
        return None

    @classmethod
    def find_n_m_pdiv(cls, fin, fout, max_pdiv, exclude=(), debug=0):
        """Find integer PLL values pdiv, n and m.
    
        :param fin: PLL input frequency reference (Hz)
        :param fout: PLL output frequency (Hz)
        :param max_pdiv:
        :param exclude: excluded frequencies
        :param debug: Debug level
    
        :return: List consisting of
            n, m, pdiv -- values needed to compute p, q, r register fields
            error      -- frequency error (Hz)
        """
    
        # CDCE9xx formula from section 9.2.2.2 of CDCE913 data sheet
        #
        # fout = (fin / pdiv) * (n / m)
        #
        # fvco = fin * n / m
    
        best = (None, None, None, None, None)
        min_error = fout
    
        for pdiv in cls.pdivs(fout, max_pdiv):
            fvco = pdiv * fout
    
            # Check for harmonics of fvco in GPS band
            band = harmonic_in_band(fvco, exclude)
            if band is not None:
                # Skip this pdiv, because band includes harmonic of fvco
                if debug > 1:
                    print('harmonic of: PDIV=%04d FVCO=%.2e in: %s' % (pdiv,
                        fvco, band), file=sys.stderr)
                continue
    
            # Calculate maximum possible M for fvco
            max_m = round(fin * cls.MAX_N / fvco)
            if max_m > cls.MAX_M:
                max_m = cls.MAX_M
    
            # Find N that produces minimum error starting at highest M
            for m in range(max_m, cls.MIN_M, -1):
    
                # Find nearest integer value for n
                n = round(fvco * m / fin)
    
                # Calculate actual output frequency and error
                error = fin * n / (pdiv * m) - fout
                abs_error = abs(error)
                #if args.d:
    
                if error == 0:
                    return(n, m, pdiv, fvco, 0)
    
                if abs_error < min_error:
                    best = (n, m, pdiv, fvco, error)
                    if debug > 1:
                        print('N=%04d M=%03d PDIV=%04d FVCO=%.2e ERROR=%+.4f'
                               % best, file=sys.stderr) # debug
                    min_error = abs_error
    
        return best
    
    @classmethod
    def calculate_p_q_r(cls, n, m):
        """Calculate values of p, q, r fields used in CDCE9xx registers."""

        p = 4 - int(math.log2(n/m))
        n_prime = n * 2**p
        q = int(n_prime/m)
        r = n_prime - m*q
    
        # Check results
        if ((cls.MIN_P <= p <= cls.MAX_P) and
            (cls.MIN_Q <= q <= cls.MAX_Q) and
            (cls.MIN_R <= r <= cls.MAX_R)):
               return(p, q, r)
    
        return(None, None, None)

    @classmethod
    def config_fields(cls, n_plls):
        for i in range(1, n_plls + 1):
            for j in cls.register(i):
                yield j

    @classmethod
    def calculate_m(cls, n, p, q, r):
        '''Calculate and return m from n, p, q, r register values.'''

        # Don't check maximum limits as these fields are limited by register size
        if  cls.MIN_Q <= q:
            return (n * 2**p - r) // q
        else:
            print('%s: error: Q=%d <%d' % (__name__, q, cls.MIN_Q), file=sys.stderr)
            return None

    @classmethod
    def calculate_fvco_fout(cls, fin, n, m, pdiv):
        '''Calculate and return fvco and fout from fin, N, M, PDIV.'''

        if n >= m:
            # Check fvco
            fvco = fin * n / m
           #if cls.vco_range(fvco) is None:
            if False:
                print('%s: error: fvco=%d out of range' % (__name__, fvco),
                       file=sys.stderr)
                return (None, None)

            else:
                if pdiv and m:
                    return (fvco, fvco / pdiv)
                else:
                    return (0.0, 0.0)  # No output if PDIV=0 TODO test m=0
        else:
            print('%s: error: N=%d <%d' % (__name__, n, m),
                   file=sys.stderr)
            return None

    def register(pll_n):
        """Yield fields in PLL with index pll_n."""

        p_base = pll_n * 0x10
        m_base = pll_n * 2
        YY = 'Y%dY%d' % (m_base, m_base + 1)
    
        # Descriptions of the PLL fields
        # r_seq, w_seq: priority for read and write operations; used to sequence multiple commands
        # d_val: default value
        # bit_offset: number of bits to right of field in register
        # bit_size: size of field in bits
        FIELDS = (                  # (r_seq, w_seq, d_val, (reg_addr, bit_offset, bit_size))
            ('SCC%d_7' % pll_n,        Field(50, 50, 0,     [Spec(p_base + 0, 5, 3)])),
            ('SCC%d_6' % pll_n,        Field(50, 50, 0,     [Spec(p_base + 0, 2, 3)])),
            ('SCC%d_5' % pll_n,        Field(50, 50, 0,     [Spec(p_base + 0, 2, 2),
                                                             Spec(p_base + 1, 7, 1)])),
            ('SCC%d_4' % pll_n,        Field(50, 50, 0,     [Spec(p_base + 1, 4, 3)])),
            ('SCC%d_3' % pll_n,        Field(50, 50, 0,     [Spec(p_base + 1, 1, 3)])),
            ('SCC%d_2' % pll_n,        Field(50, 50, 0,     [Spec(p_base + 1, 0, 1),
                                                             Spec(p_base + 2, 6, 2)])),
            ('SCC%d_1' % pll_n,        Field(50, 50, 0,     [Spec(p_base + 2, 3, 3)])),
            ('SCC%d_0' % pll_n,        Field(50, 50, 0,     [Spec(p_base + 2, 0, 3)])),
    
        ) + tuple(
            ('FS%d_%d' % (pll_n, s),   Field(50, 50, 0,     [Spec(p_base + 3, s, 1)]))
                                       for s in range(0, 8)) + (
        
            ('MUX%d'   % pll_n,        Field(50, 50, 1,     [Spec(p_base + 4, 7, 1)])),
            ('M%d'     % m_base,       Field(50, 50, 1,     [Spec(p_base + 4, 6, 1)])),
            ('M%d'     % (m_base + 1), Field(50, 50, 2,     [Spec(p_base + 4, 4, 2)])),
            (YY + '_ST1',              Field(50, 50, 3,     [Spec(p_base + 4, 2, 2)])),
            (YY + '_ST0',              Field(50, 50, 1,     [Spec(p_base + 4, 0, 2)])),
    
            (YY + '_0',                Field(50, 50, 0,     [Spec(p_base + 5, 0, 0)])),
            (YY + '_1',                Field(50, 50, 1,     [Spec(p_base + 5, 1, 1)])),
        ) + tuple(
            (YY + '_%d' % s,           Field(50, 50, 0,     [Spec(p_base + 5, s, 1)]))
                                       for s in range(2, 8)) + (
    
            ('SCC%dDC' % pll_n,        Field(50, 50, 0,     [Spec(p_base + 6, 6, 1)])),
            ('PDIV%d'  % m_base,       Field(50, 50, 1,     [Spec(p_base + 6, 0, 7)])),
    
            ('_%X_6' % (p_base + 7),   Field(0,   0, 0,     [Spec(p_base + 7, 6, 1)])),
            ('PDIV%d' % (m_base + 1),  Field(50, 50, 1,     [Spec(p_base + 7, 0, 7)])),
    
            ('PLL%d_0N' % pll_n,       Field(50, 50, 0x4,   [Spec(p_base + 8, 0, 8),
                                                             Spec(p_base + 9, 4, 4)])),
            ('PLL%d_0R' % pll_n,       Field(50, 50, 0x0,   [Spec(p_base + 9, 0, 4),
                                                             Spec(p_base + 0xA, 3, 5)])),
            ('PLL%d_0Q' % pll_n,       Field(50, 50, 0x10,  [Spec(p_base + 0xA, 0, 3),
                                                             Spec(p_base + 0xB, 5, 3)])),
            ('PLL%d_0P' % pll_n,       Field(50, 50, 0x2,   [Spec(p_base + 0xB, 2, 3)])),
            ('VCO%d_0_RANGE' % pll_n ,
                                       Field(50, 50, 0,     [Spec(p_base + 0xB, 0, 2)])),
    
            ('PLL%d_1N' % pll_n,       Field(50, 50, 0x4,   [Spec(p_base + 0xC, 0, 8),
                                                             Spec(p_base + 0xD, 4, 4)])),
            ('PLL%d_1R' % pll_n,       Field(50, 50, 0,     [Spec(p_base + 0xD, 0, 4),
                                                             Spec(p_base + 0xE, 3, 5)])),
            ('PLL%d_1Q' % pll_n,       Field(50, 50, 0x10,  [Spec(p_base + 0xE, 0, 3),
                                                             Spec(p_base + 0xF, 5, 3)])),
            ('PLL%d_1P' % pll_n,       Field(50, 50, 2,     [Spec(p_base + 0xF, 2, 3)])),
            ('VCO%d_1_RANGE' % pll_n,
                                       Field(50, 50, 0,     [Spec(p_base + 0xF, 0, 2)])),
        )
    
        for i in FIELDS:
            yield i

class CDCE9xx:
    '''Class for a Texas Instruments CDCE9xx Clock Generator.'''

    DEF_PLL = 1
    DEF_PDIV = 1

    # CDCE9xx Constants
    MIN_FIN = 8e6  # Hz
    MAX_FIN = 32e6

    MIN_FOUT = int(PLL.MIN_VCO / PLL.MAX_PDIV1)
    MAX_FOUT = int(PLL.MAX_VCO)
    
    BASE_ADDR = 0x64
    BYTE_OP = 0x80  # Byte read or write

    MAX_EEPIP_LOOPS = 255

    # Configuration Register Fields
    FIELDS = (      # (r_seq, w_seq, d_val, (reg_addr, bit_offset, bit_size))
        ('E_EL',       Field(50,  0,    0, [Spec(0x0, 7, 1)])),  # Actually for CDCEL9xx
        ('RID',        Field(50,  0,    0, [Spec(0x0, 4, 3)])),
        ('VID',        Field(50,  0,    1, [Spec(0x0, 0, 4)])),
    
        ('_1_7',       Field( 0,  0,    0, [Spec(0x1, 7, 1)])),  # Reserved
        ('EEPIP',      Field(50,  0,    0, [Spec(0x1, 6, 1)])),
        ('EELOCK',     Field(50, 70,    0, [Spec(0x1, 5, 1)])),  # Permanently locks EEPROM
        ('PWDN',       Field(50, 50,    0, [Spec(0x1, 4, 1)])),
        ('INCLK',      Field(50, 50,    0, [Spec(0x1, 2, 2)])),
        ('SLAVE_ADDR', Field(50, 80, None, [Spec(0x1, 0, 2)])),  # Varies within CDCE9xx family
    
        ('M1',         Field(50, 50,    1, [Spec(0x2, 7, 1)])),
        ('SPICON',     Field(50, 90,    0, [Spec(0x2, 6, 1)])),  # Turns off I2C interface
        ('Y1_ST1',     Field(50, 50,    3, [Spec(0x2, 4, 2)])),
        ('Y1_ST0',     Field(50, 50,    1, [Spec(0x2, 2, 2)])),
        ('PDIV1',      Field(50, 50,    1, [Spec(0x2, 0, 2),

                                            Spec(0x3, 0, 8)])),
        ('Y1_0',       Field(50, 50,    0, [Spec(0x4, 0, 1)])),
        ('Y1_1',       Field(50, 50,    1, [Spec(0x4, 1, 1)])),
    ) + tuple(
        ('Y1_%d' % n,  Field(50, 50,    0, [Spec(0x4, n, 1)])) for n in range(2, 8)
    ) + (
        ('XCSEL',      Field(50, 50,   10, [Spec(0x5, 3, 5)])),
        ('_5_0',       Field( 0,  0,    0, [Spec(0x5, 0, 3)])),  # Reserved
        
        ('BCOUNT',     Field(50, 50, None, [Spec(0x6, 1, 7)])),  # BCOUNT varies within CDCE9 family
         # EEPROM write cycles: MIN=100, TYP=1000
         # EELOCK and EEPROM are sequenced so that two separate sets are required to lock EEPROM
        ('EEWRITE',    Field(50, 60,    0, [Spec(0x6, 0, 1)])),
    ) + tuple(
        ('_%X_0' % n,  Field( 0,  0,    0, [Spec(n, 0, 8)])) for n in range(7, 0x10)
    )

    # Functions
    @classmethod
    def def_addr(cls, n_plls):
        """Return base address of a CDCE9xx with n_plls."""

        return cls.BASE_ADDR + (((n_plls - 1) & 2) << 2) + (n_plls & 1)

    @classmethod
    def fin_valid(cls, f):
        '''Test if f is valid input frequency.'''
    
        if cls.MIN_FIN <= f <= cls.MAX_FIN:
            return True
    
        return False
    
    def __init__(self, n_plls, port, addr):
        '''Construct a clock generator object.'''

        # Build config_fields common to all PLLs
        self.config_fields = OrderedDict(CDCE9xx.FIELDS)
        self.config_fields['BCOUNT'].d_val = 0x10 * (1 + n_plls)
        self.config_fields['SLAVE_ADDR'].d_val = n_plls & 1

        # Build PLL-specific config_fields
        self.config_fields.update(PLL.config_fields(n_plls))

        self.port = port  # i2c bus (smbus) object
        self.addr = addr  # address of device on bus

    def set_pll_pdiv(self, pll_n, n, m, vco, pdiv_n, pdiv, state_n=0, debug=0):
        '''Given fin, program PLL and PDIV to generate fout.
        Return (frequency error) on success else None.'''

        # Calculate and validate register values
        p = None
        p, q, r = PLL.calculate_p_q_r(n, m)

        if p is None:
            return False
        else:
            if pll_n:
                if pdiv_n == 0:
                    if pll_n == 1:
                        pdiv_n = 1
                    else:
                        #pdiv_n = (pll_n * 2) + 1  # TODO for next board rev
                        pdiv_n = (pll_n * 2)

                state = self.get('PWDN')

                return self.set_fields(
                    (
                        ('PWDN', 1),  # Prevent spurious output during update
                        ('PLL%d_%dN' % (pll_n, state_n), n),
                        ('PLL%d_%dP' % (pll_n, state_n), p),
                        ('PLL%d_%dQ' % (pll_n, state_n), q),
                        ('PLL%d_%dR' % (pll_n, state_n), r),
                        ('VCO%d_%d_RANGE' % (pll_n, state_n), vco),
                        ('MUX%d' % pll_n, 0),
                        ('PDIV%d' % pdiv_n, pdiv),
                        ('PWDN', state),
                    ), debug
                )

            else:
                return True

    def get(self, name):
        """Get value of field.

        Arguments:
            name     -- string containing field's name

        Returns: value of field
        """

        value = 0
        for spec in self.config_fields[name].specs:
            mask = (1 << spec.size) - 1

            # Get PLL register
            value = (value << spec.size) | (mask & (self.port.read_byte_data(
                                                        self.addr,
                                                        self.BYTE_OP|spec.reg) >> spec.offset)
                                           )
        return value

    def get_fields(self, names, debug=0):
        '''Returns a list of (field_name, value) tuples from PLL.'''

        # Build a list of readable (name, value) tuples
        work = []
        for name in names:
            if name in self.config_fields:
                seq = self.config_fields[name].r_seq

                if seq:  # Skip read-protected fields
                    work += [(seq, name)]
                else:
                    if debug:
                        print('%s: warning: %s: read-protected' % (__name__, name),
                               file=sys.stderr)
            else:
                if debug:
                    print('%s: error: %s: unknown field' % (__name__, name),
                           file=sys.stderr)

        # Sort work for increasing r_seq
        work = sorted(work, key=lambda item: item[0])

        result = []
        for _, name in work:
            result += [(name, self.get(name))]

        return result

    def set_fields(self, names_values, debug=0):
        '''Updates valid fields using (name, value) tuples in a tuple or list.
        Returns number of fields processed.'''

        work = []
        # Build list of valid names
        for name, value in names_values:
            if name in self.config_fields:
                seq = self.config_fields[name].w_seq

                if seq:  # Skip write-protected fields
                    work += [(seq, name, value)]
                else:
                    if debug:
                        print('%s: warning: %s: write-protected' % (__name__, name),
                               file=sys.stderr)
            else:
                if debug:          
                    print('%s: error: %s: unknown field' % (__name__, name),
                           file=sys.stderr)

        # Sort work for increasing w_seq
        work = sorted(work, key=lambda item: item[0])

        for _, name, value in work:
            self.set(name, value)

        return len(work)

    def set(self, name, value):
        """Set field with name to value.

        Arguments:
            name     -- string containing field's name
            value    -- value to write to field
        """

        # write least significant bits first
        for spec in reversed(self.config_fields[name].specs):
            value = self.poke(spec, value)

        return

    def poke(self, spec, value):
        '''Peel LS bits out of value into bit field defined by spec.'''

        mask = (1 << spec.size) - 1

        # Get register value from PLL
        reg_val = self.port.read_byte_data(self.addr, self.BYTE_OP|spec.reg)

        # Update field
        reg_val &= ~(mask << spec.offset)
        self.port.write_byte_data(self.addr, self.BYTE_OP|spec.reg,
                                  reg_val | ((value & mask) << spec.offset))

        # return unprocessed bits
        return value >> spec.size 

    def dump(self, zeroes):
        '''Dump device configuration as JSON.'''

        names_values = self.get_fields(list(self.config_fields))

        if not zeroes:
            names_values = [n_v for n_v in names_values if n_v[1]]

        print_json(names_values)
        return

    def load(self, fp):
        '''Load device configuration from JSON.'''

        j = json.load(fp)
        names_values = list(j.items())
        self.set_fields(names_values)
        return

    def write_eeprom(self):
        """Write current configuration to EEPROM, wait for completion."""

        self.set('EEWRITE', 1)

        i = self.MAX_EEPIP_LOOPS
        while i:
            i -= 1

            if not self.get('EEPIP'):
                self.set('EEWRITE', 0)
                return self.MAX_EEPIP_LOOPS - i  # Number of gets

        return False

    def get_nmp(self, pll_n=1, pdiv_n=1, state_n=0):
        """Return N, M, P of pll_n, pdiv_n, state in a list."""

        n = self.get('PLL%d_%dN' % (pll_n, state_n))

        return (n,
                PLL.calculate_m(n,
                            self.get('PLL%d_%dP' % (pll_n, state_n)),
                            self.get('PLL%d_%dQ' % (pll_n, state_n)),
                            self.get('PLL%d_%dR' % (pll_n, state_n))
                ),
                self.get('PDIV%d' % (pdiv_n))
        )

    def factory_default(self):
        """Return board defaults for all fields."""

        return self.set_fields([(n, self.config_fields[n].d_val) for n in self.config_fields])

if __name__ == "__main__":
    pass
