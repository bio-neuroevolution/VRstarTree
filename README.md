# VRstarTree
A implemention of Verkel Adptive R*-tree(Verkle AR*-Tree)

The website contains the source code and data in the paper 《Adaptive Spatio-temporal Query Strategies in Blockchain》. </br>
Among them, we have implemented a adaptive verkel R*-tree(Verkle AR*-tree), which combines verkle vector commitment and R*-tree index to support the storage of spatiotemporal data in blockchain.</br>
We further improve R*-tree, which has the ability of query adaptation and supports more efficient spatio-temporal query.</br>

## Directory "pRTree" contains the implementation of AR*-tree
## Directory "pVerkle" contains the verkle proof and the implementation of blockchain, in which the code of Verkle proof comes from https://github.com/coinstudent2048/verklebp.</br> We modifiy it for  spatio-temporal index tree.
##  Directory "spatiotemporal_blockdag" is the code of Merkle KD-tree. We used for the comparative experiment. The code is quoted from https://github.com/ILDAR9/spatiotemporal_blockdag </br> and has made two changes: the original KD tree does not support the storage of cube objects and the query of time dimension. We have added these two functions.
## Directory "experiment" includes four experiments under e1, e2, e3 and e4 subfolders.
### Experiment 1 realized the access performance comparison of verkle R*-tree, verkle AR*-tree and Merkle KD tree based on pokeman dataset.
### Experiment 2 changed the pokeman data set and changed part of the location data into a spatiotemporal cube, and repeat experiment 1 on the new data set
### Experiment 3 compares the performance of adaptive algorithm and non adaptive algorithm under different R*-tree parameters.
### Experiment 4 compared the proof lengths of verkle AR*-tree and Merkel KD tree, as well as verkle tree and MPT.



