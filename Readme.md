# DCI-SIMPY

DCI-SIMPY is a simulation of a distributed architecture for event processing using SimPy.  
The project mimics the behavior of a model developed in Salabim, where each node in the hierarchy, has a limited number of "observation windows" to process events. When the local load exceeds the available capacity, events are offloaded to an upper-level node.

## Project Structure
DCI-SIMPY/ 
 └── sim/ 
    └── scenario.py # Main simulation script using SimPy
├── config.yaml # Simulation configuration file (YAML format) 
├── README.md # Project documentation and instructions
├── graphs/ # Folder where generated graphs are stored 
 

## Requirements

- Python 3.x
- SimPy
- PyYAML
- Pandas
- Matplotlib
- NumPy

You can install the required dependencies using pip:

```bash
pip install simpy pyyaml pandas matplotlib numpy

python sim/scenario.py