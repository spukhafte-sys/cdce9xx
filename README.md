# TI CDCE9XX Clock Generator Driver and Application for Python3

This Python3 command line tool and library controls any of the Texas Instruments
CDCE9xx family of programmable spread-spectrum clock generators on an I2C bus:
* CDCE913/CDCEL913: 1 PLL, 3 Outputs
* CDCE925/CDCEL925: 2 PLLs, 5 Outputs
* CDCE937/CDCEL937: 3 PLLs, 7 Outputs
* CDCE949/CDCEL949: 4 PLLs, 9 Outputs

## Features and Benefits
* One class instatiates objects for entire family of devices
* Calculates PLL and PDIV configurations given input and desired output frequencies
* Excludes PLL configurations with VCO harmonics in sensitive bands (like GPS)
* Configure individual registers by name
* Export and import configurations in JSON format
* Detects platform and uses appropriate I2C bus

## Dependencies
Requires:
* Python >= 3.7
* smbus >= 1.1
* Adafruit-PlatformDetect >= 3.19.3

## Installation
```bash
pip3 install spukhafte-cdce9xx
```

## Usage
Command line tool:
```bash
cdce9xx -h
```
or in a Python3 program:
```bash
from spukhafte.cdce9xx import CDCE9XX
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss
what you would like to change.

Please update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
