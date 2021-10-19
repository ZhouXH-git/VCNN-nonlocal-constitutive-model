#!/bin/bash
# This is a script for generation a family of flows.

echo 'generation began!'

for i in $(seq 1.0 0.05 2.01); # (start step stop)
do

  cp -r flow_template flow_$i
  new="aw $i;"
  sed -i "19c $new" flow_$i/system/blockMeshDict
  cd flow_$i
  blockMesh
  simpleFoam > log.out
  cd ..
  
  cp -r transport_template transport_$i
  cp flow_$i/3000/U transport_$i/0        # when aw in (0.5,2.0)->2000; when aw in (2.0,4.0)->3000
  cp flow_$i/system/blockMeshDict transport_$i/system/blockMeshDict
  cd transport_$i
  blockMesh
  checkMesh -writeFields "(wallDistance)"
  myTFoam > log.out
  postProcess -func writeCellCentres
  postProcess -func writeCellVolumes
  cd ..
   
done

echo 'generation finished!'








