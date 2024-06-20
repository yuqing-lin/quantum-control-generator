# Quantum Control Generator

Quantum Control Generator is a tool for creating and managing control signals for transmon-cavity quantum systems. 

## Features

- Drag and drop interface
- Control signal visualization
- Support for SNAP, displacement, and ECD gates

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yuqing-lin/quantum-control-generator.git
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```
python main.py
```
This will launch the GUI, where you can drag and drop gates to assemble the circuit, generate the corresponding control pulses, and save them to a CSV file.

## References

1. Heeres, R. W., Vlastakis, B., Holland, E., Krastanov, S., Albert, V. V., Frunzio, L., Jiang, L., & Schoelkopf, R. J. (2015). Cavity state manipulation using photon-number selective phase gates. *Physical Review Letters, 115*(13), 137002. [https://doi.org/10.1103/PhysRevLett.115.137002](https://doi.org/10.1103/PhysRevLett.115.137002)

2. Eickbusch, A., Sivak, V., Ding, A. Z., Elder, S. S., Jha, S. R., Venkatraman, J., Royer, B., Girvin, S. M., Schoelkopf, R. J., & Devoret, M. H. (2022). Fast universal control of an oscillator with weak dispersive coupling to a qubit. *Nature Physics, 18*(12), 1464-1469. [https://doi.org/10.1038/s41567-022-01691-2](https://doi.org/10.1038/s41567-022-01691-2)

## Acknowledgments

- This project uses code for ECD pulse generation adapted from [Echoed Conditional Displacement (ECD) Control](https://github.com/alec-eickbusch/ECD_control).