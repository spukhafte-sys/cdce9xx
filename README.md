# TI CDCE9XX Clock Generator Driver and Application for Python3

This Python3 command line tool and library controls any of the Texas Instruments
CDCE9xx family of programmable spread-spectrum clock generators on an I2C bus:
* CDCE913/CDCEL913: 1 PLL, 3 Outputs
* CDCE925/CDCEL925: 2 PLLs, 5 Outputs
* CDCE937/CDCEL937: 3 PLLs, 7 Outputs
* CDCE949/CDCEL949: 4 PLLs, 9 Outputs

## Dependencies
Requires:
* Python>=3.7
* smbus>=1.1
* Adafruit-PlatformDetect>=3.19.3

## Installation
```bash
pip3 install ti-cdce9xx
```

## Usage
```bash
cdce9xx
```
or
```bash
from cdce9xx import CDCE9XX
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss
what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
