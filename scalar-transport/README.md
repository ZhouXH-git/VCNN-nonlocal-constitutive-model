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

./
get_fullfield.py # (foam_synthetic_truth, 5000, .)
rm UyFullField UzFullField
get_point.py # (foam_synthetic_truth, UxFullField, 0.05, 0.25, 0.005, UxPoint_0)

postProcess -func writeCellCentres -case foam_synthetic_truth -time '5000'
cp foam_synthetic_truth/5000/Cy ./y # (manually delete header/end, leave only the 50 values)

get_inputs.py # (foam_synthetic_truth, 5000, .)
```

