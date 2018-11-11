# Generative Neural Networks for Randomized Optimization.

This code is developed to experiments with invertible Generative Neural Networks in a randomized optimization setting, as proposed in ??.

## Code organization
The code is organized as follows:

- **run_exp.py**: runs an optimization procedure given an experiment configurations.
- **plot.py**: plotting utilities.
- **models**: defines the different models (MLP, invertible MLP, ..)
- **distributions**: construct distributions (Real NVP, NICE, DDM, ..).
- **objectives**: define the different objective functions.
- **algorithms**: implements different ES algorithms (VSG, aVSG, TRSG, NSG).


## Dependencies and requirements
The code was run using Python 3.6.4 and Tensorflow 1.10.0.
You will need matplotlib, numpy and scipy packages as well. 

## Running the code
### Setting up the environment
You can follow the [TensorFlow installation guide](https://www.tensorflow.org/install/) to set up a virtualenv in which you can run the code.
On a MacOS, running the following commands inside the directory should get you a working environment:

	virtualenv -p python3 myvirtualenv
	source myvirtualenv/bin/activate
	pip3 install tensorflow
	pip3 install matplotlib
	pip3 install scipy


### Providing a config
The script **run_exp.py** takes two input arguments: the path to a configuration (a json file describing which distribution to train via which algorithms on which objective function) and the path to the output logs. 

Some default config files can be found in the `config` folder. You can make sure your installation is working by running the following command (after activation of your virtual environment):

	python run_exp.py --config_name vsg_sphere_gaussian.json --output_name vsg_sphere_gaussian_logs.json
	
which trains a normal distribution on the sphere function via Vanilla Search Gradient. The results will be stored in the `logs`directory and can be plotted thanks to the script **plot.py**, which by default plots performances and analysis (entropy, surrogate loss) for all log files in the `logs` directory.


## License

Copyright CRITEO

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.