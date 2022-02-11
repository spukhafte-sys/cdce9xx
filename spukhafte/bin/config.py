#!/usr/bin/env python3
"""Configure the CDCE913 PLL in the K15

   Requirements
   python (>=3.5)
   smbus (1.1.post2)
   cdce9xx
"""

__version__ = '0.0'
__author__ = 'Fred Fierling'
__copyright__ = 'Copyright 2019, Spukhafte Systems Limited'

import os
import sys
import argparse

from smbus import SMBus

from spukhafte.cdce9xx import harmonic_in_band, CDCE9xx, PLL
from adafruit_platformdetect import Detector

# Error return values
SUCCESS = 0
FIN_ERROR = 1
FOUT_ERROR = 2
PDIV_ERROR = 3
SOLUTION_ERROR = 4
SET_ERROR = 5
VCO_ERROR = 6
PLL_ERROR = 7

LANGUAGE = 'english'  # TODO international support

# GPS band spec
# (center_freq, span) in Hz
GPS_BANDS = (
        (1567.74750e6, 1583.09250e6, 'GPS L1C/A'),  # Subset of E1B/C
#       (1563.42000e6, 1587.42000e6, 'QZSS'),
#       (1559.00000e6, 1591.00000e6, 'Galileo E1B/C'),
#       (1589.06250e6, 1605.37500e6, 'GLONASS L1OF'),
#       (1559.05200e6, 1591.78800e6, 'BeiDou B1'),
)

# Functions
def auto_int(x):
    return int(x, 0)

def print_factors(prefix, product):
    modf, whole = utilities.math.modf(product)
    out = ''
    for b, e in utilities.factorize(int(whole)):
        if out is '':
            out += prefix
        else:
            out += "*"

        if e == 1:
            out += "%d" % (b)
        else:
            out += "%d^%d" % (b, e)

    if modf:
        out += '%+.4f' % modf

    print(out, file=sys.stderr)

def main(device=None):
    # Board-specific defaults
    # Policy here is to set unused dividers to zero (0Hz out)
    # and multiplex dividers straight to their respective outputs.
    # Eg, PDIV2 -> Y2
    # VALID_PLLS: Valid PLL indices (-p arguments); 0 means null PLL, used for setting only PDIV
    # DEF_PDIV_MAP: List that maps PLL index to index of its default PDIV; same len() as VALID_PLLS
    # VALID_PDIVS: PDIV indices (-d arguments) connected on the board

    vendor, model = (device.upper()).split('_')
    if model.startswith('K'):
        if model == 'K15' or model == 'K25':
            DEF_FIN = 30.72e6  # u-blox LEA-M8F REF_FREQ_OUT
            DEF_CONFIG = (
                ("INCLK", 2),  # LVCMOS
                ("M2", 0),  # R1: PDIV1
               #("M3", 0),  # R2: PDIV1
                ("PDIV2", 0), # Reset and stand-by
                ("PDIV3", 0), # Reset and stand-by
            )  # TODO check for R2 board

            EXCLUDE = GPS_BANDS
            N_PLLS = 1
           #DEF_PDIV_MAP = (1, 3)
           #VALID_PDIVS = (1, 2)  # R1: Y3 connects to a test point
            VALID_PLLS = (0, 1)
            DEF_PDIV_MAP = (1, 1)
            VALID_PDIVS = (1, 3)  # R2: Y2 connects to a test point

        else: 
            DEF_FIN = 27e6  # XTAL
            DEF_CONFIG = (
                    (("MUX1", 0),
                    ("Y1_ST0", 3),
                    ("XCSEL", 5)) +  # Trim oscillator R0
                    tuple(("PDIV%d" % i, 0) for i in range(1,8))  # Reset and stand-by
                    )

            EXCLUDE = ()
            N_PLLS = 3
           #DEF_PDIV_MAP = (1, 1, 4) # R1
           #VALID_PDIVS = (1, 2, 4)  # R1: Y3, Y5 connect to test points
            VALID_PLLS = (0, 1, 2, 3)  # Zero means null pll
            DEF_PDIV_MAP = (1, 1, 5, 7)
            VALID_PDIVS = (1, 3, 4, 5, 7)  # R2: Y2, Y6 connect to test points
    else:
        DEF_FIN = 27e6  # XTAL
        DEF_CONFIG = (
            ("PDIV1", 0), # Reset and stand-by
            ("PDIV2", 0), # Reset and stand-by
            ("PDIV3", 0), # Reset and stand-by
        )  #

        EXCLUDE = ()
        N_PLLS = 1
        VALID_PLLS = (0, 1, 2)
        DEF_PDIV_MAP = (1, 1, 1)
        VALID_PDIVS = (1, 2, 3)  # R2: Y2 connects to a test point

    # Environment-specific defaults
    HW = Detector()

    if HW.board.id == 'BEAGLEBONE_BLACK':
        VALID_BUSES = (0,1,2)
        DEF_BUS = 1 if model == 'K25' else 2
    elif HW.board.id.startswith('RASPBERRY_PI'):
        VALID_BUSES = (0,1,2)
        DEF_BUS = 1
    elif HW.board.id == 'BEAGLEBONE_AI':
        VALID_BUSES = (0,3)
        DEF_BUS = 3
    else:
        VALID_BUSES = (0,1,2)
        DEF_BUS = 0

    ENV_PREFIX = vendor + '_' + model

    var = ENV_PREFIX + '_BUS'
    if var in os.environ:
        DEF_FIN = auto_int(os.environ[var])

    var = ENV_PREFIX + '_ADDR'
    if var in os.environ:
        DEF_ADDR = auto_int(os.environ[var])
    else:
        DEF_ADDR = CDCE9xx.def_addr(N_PLLS)

    var = ENV_PREFIX + '_FIN'
    if var in os.environ:
        DEF_FIN = float(os.environ[var])

# Mainline ==========================
    parser = argparse.ArgumentParser(
        description=("configure %s's PLL" % model),
        epilog='no arguments: list PLL1 configuration')

    group = parser.add_mutually_exclusive_group()

    # Mutually exclusive arguments
    group.add_argument('fout', type=float, nargs='?',
        help="output frequency (Hz)",
        metavar='FOUT'
    )

    group.add_argument('-nmp', type=int, nargs=3,
        metavar=('N', 'M', 'PDIV'),
        help="where FOUT = N/M * FIN/PDIV"
    )

    # Optional arguments
    parser.add_argument('-a', dest='addr', type=auto_int, default=DEF_ADDR,
        help="I2C address of CDCE9%d%d (default=0x%x)"
              % (N_PLLS, (1 + 2 * N_PLLS), DEF_ADDR))

    parser.add_argument('-b', dest='bus', type=int, choices=VALID_BUSES,
        default=DEF_BUS,
        help="I2C bus number (default=%d)" % (DEF_BUS))

    parser.add_argument('-d', type=int, default=None,
        choices=VALID_PDIVS,
       #metavar=('PDIV'),
        help="target PDIV index")

    parser.add_argument('-fin', dest='fin', type=float, default=DEF_FIN,
        help="input frequency Xin/CLK, (default=%.2f MHz)" % (DEF_FIN/1e6))

    parser.add_argument('-g', action="store_true", 
        help="allow VCO harmonics in GPS bands")

    parser.add_argument('-i', type=argparse.FileType('r'),
        metavar=('FILE'),
        help='file containing config JSON')

    parser.add_argument('-j', action="store_true", 
        help="output configuration in JSON")

    parser.add_argument('-k', dest='pdiv10', action="store_true", default=False,
        help="solve for a ten bit divider")

    parser.add_argument('-p', type=int, default=CDCE9xx.DEF_PLL,
        choices=VALID_PLLS,
       #metavar=('PLL'),
        help="target PLL index (default=%d, 0: null device)" % CDCE9xx.DEF_PLL)

    parser.add_argument('-r', action="store_true", 
        help="reset to factory default")

    parser.add_argument('-s', type=int, default=0, choices=(0, 1),
        help="target PLL state, (default=%(default)s)")

    parser.add_argument('-w', action="store_true", help="write EEPROM")
    parser.add_argument('-v', default=0, action="count", help="verbosity level")
    parser.add_argument('-z', action="store_true",
        help="dump zero value config JSON")
 
    args = parser.parse_args()
 
    # Check fin
    if not CDCE9xx.fin_valid(args.fin):
        parser.error('fin out of range')
        return(FIN_ERROR)

    # Check fout
    if args.fout and not PLL.fout_valid(args.fout, args.pdiv10):
        parser.error('fout out of range')
        return(FOUT_ERROR)

    # Set target divider of PLL
    if args.d is None:
        args.d = DEF_PDIV_MAP[args.p]

    # Check size of divider
    if args.pdiv10:
        if args.d in (0, 1):
            max_pdiv = PLL.MAX_PDIV1
        else:
            parser.error('-10b and -d=%d not allowed' % args.d)
            return(PDIV_ERROR)
    else:
        max_pdiv = PLL.MAX_PDIV

    if args.v:
        print_factors('FIN=', args.fin)
        
        if args.fout:
            print_factors('FOUT=', args.fout)  

    if args.g:
        EXCLUDE = ()

    # if not args.n:
    i2c = SMBus(args.bus)
    i2c.open(args.bus)

    clock_gen = CDCE9xx(N_PLLS, i2c, args.addr)

    if args.r:
        # Set all fields to factory defaults
        clock_gen.factory_default()
        clock_gen.set_fields(DEF_CONFIG)

    if args.fout or args.nmp:
        if args.fout:
            n, m, pdiv, fvco, error = PLL.find_n_m_pdiv(args.fin, args.fout, max_pdiv,
                                              exclude=EXCLUDE, debug=args.v)
            if None in (n, m, pdiv):
                parser.error('solution not found for FIN=%.2e FOUT=%.2e' 
                        % (args.fin, args.fout))
                return(SOLUTION_ERROR)

        elif args.nmp:
            if PLL.nmp_valid(args.nmp, args.pdiv10):
                n, m, pdiv = args.nmp
            else:
                parser.error('N, M, or PDIV out of range')
                return(SET_ERROR)

            fvco = (args.fin * n) / m
            band = harmonic_in_band(fvco, EXCLUDE)

            if band is not None:
                parser.error('harmonic of FVCO=%.2e in: %s' % (fvco, band))

        vco = PLL.vco_range(fvco)
        if vco is None:
            print('%s: FVCO=%.2e out of range' % (__name__, fvco), file=sys.stderr)
            return(VCO_ERROR)

        if clock_gen.set_pll_pdiv(args.p, n, m, vco, args.d, pdiv,
                                state_n=args.s, debug=args.v):
            if args.v:
                format = 'N=%04d M=%03d PDIV=%04d FVCO=%.2e '

                # Distinguish perfect solution
                format += 'ERROR=%f' if error else 'ERROR=%d'
                print(format % (n, m, pdiv, fvco, error), file=sys.stderr)
        else:
            print('%s: error: set failed' % (__name__), file=sys.stderr)
            return(SET_ERROR)
    else:
        if args.p:
            n, m, pdiv = clock_gen.get_nmp(args.p, args.d, args.s)
        else:
            parser.error("can't read PLL0")
            return(PLL_ERROR)

    fvco, fout = PLL.calculate_fvco_fout(args.fin, n, m, pdiv)

    if fvco is not None:
        fmt = ('FIN=%.3e PLL%d_%d: N=%04d M=%03d PDIV%d=%04d FVCO=%.3e ' +
               ('FOUT=%d' if fout.is_integer() else 'FOUT=%.3e')
              )

        print(fmt % (args.fin, args.p, args.s, n, m, args.d, pdiv, fvco, fout),
              file=sys.stderr)

    if not sys.stdin.isatty():
        clock_gen.load(sys.stdin)

    # Configure PLL with JSON
    if args.i and args.i.readable():
        # Configure PLL using JSON on stdin
        clock_gen.load(args.i)

    if args.j:
        clock_gen.dump(args.z)

    if args.w:
        i = clock_gen.write_eeprom()
        
        if i:
            if args.v:
                print('EEPIPs=%d' % (i), file=sys.stderr)
        else:
            print('EEPROM write failed', file=sys.stderr)

    i2c.close()
 
    return(SUCCESS)

def k15():
    main(device='ssl_k15')

def k16():
    main(device='ssl_k16')

def k25():
    main(device='ssl_k25')

if __name__ == "__main__":
    main()