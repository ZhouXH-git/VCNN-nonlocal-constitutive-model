# Frame-independent vector-cloud neural network

## Code Repository for the following paper: #

X.H. Zhou, J. Han, H. Xiao. Frame-independent vector-cloud neural network for nonlocal constitutive modelling on arbitrary grids. _Computer Methods in Applied Mechanics and Engineering_, 2021.


## Directories

* __myTFoam__ : solver for scalar transport
* __flow_template__: foam template for generating a family of flow cases
* __transport_template__: foam template for generating a family of transport cases
* __flow_1__: foam flow case with hill slope = 1 (an example of flow case)
* __transport_1__: foam transport case with hill slope = 1 (an example of transport case)

## Steps to run
* 1. Compile the solver file (in OpenFOAM)
* 2. Run auto-generate_case.sh  (generate a family of flow & transport cases)
* 3. Run get-raw-data.sh  (read all the raw data from generated transport cases)
* 4. Run get-training-data.py  (process the raw data to training data)
* 5. Run Training-VCNN.ipynb  (train the neural network using PyTorch)
* 6. Run general-validation_full-stencil.sh (test the trained model in interpolation & extrapolation cases, but with full number of points within each cloud. Note that for testing, you may need to generate more cases in step 2.)

Remember:
* Change all the paths in the code to your own paths!

```sh
wmake myTFoam

./auto-generate_case.sh

./get-raw-data.sh

get-training-data.py

Training-VCNN.ipynb

./general-validation_full-stencil.sh

```

