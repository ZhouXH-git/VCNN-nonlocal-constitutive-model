* __myTFoam__ : solver for scalar transport
* __flow_template__: foam template for generating a family of flow cases
* __transport_template__: foam template for generating a family of transport cases
* __flow_1__: foam flow case with hill slope = 1 (an example of flow case)
* __transport_1__: foam transport case with hill slope = 1 (an example of transport case)

Steps to run:
* 1. Run auto-generate_case.sh  (to generate a family of flow & transport cases)
* 2. Run get-raw-data.sh  (to read all the raw data needed from generated transport cases)
* 3. Run get-training-data.py  (to get the final training data based on the raw data)
* 4. Run Training-VCNN.ipynb  (to train the neural network using the training data)

```sh
wmake myTFoam

./auto-generate_case.sh

./get-raw-data.sh

get-training-data.py

Training-VCNN.ipynb

```

