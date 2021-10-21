* __myTFoam__ : solver for scalar transport
* __flow_template__: foam template for generating a family of flow cases
* __transport_template__: foam template for generating a family of transport cases
* __flow_1__: foam flow case with hill slope = 1 (an example of flow case)
* __transport_1__: foam transport case with hill slope = 1 (an example of transport case)

Steps to run:
* 1. Compile the solver file (in OpenFOAM)
* 2. Run auto-generate_case.sh  (to generate a family of flow & transport cases)
* 3. Run get-raw-data.sh  (to read all the raw data needed from generated transport cases)
* 4. Run get-training-data.py  (to get the final training data based on the raw data)
* 5. Run Training-VCNN.ipynb  (to train the neural network using the training data with fixed number of points within each cloud)
* 6. Run general-validation_full-stencil.sh (to test the trained model in interpolation & extrapolation cases, but with full number of points within each cloud. Note that for testing, you may need to generate more cases in step 2.)

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

