# KATE - Kicking Autism Through Entertainment

KATE is a therapeutic video game for the treatment of autism through human-machine interaction. 
KATE's objective is two-fold: on the one hand, the game contributes to the social rehabilitation of children, and on the other hand, it is scientific, as it makes it possible to advance the research into the treatment of autism through video games.

## Getting Started

How to go through our analysis?

We recommend to first go thorugh the Kinect part, which was done first (since the first version of the game was for Kinect).

Starting with the Kinect analysis in the following order:

0) Optional: Data parsing file
1) Data visualization
2) K-Neighbors
3) Neural Network
4) TPOT Analysis

Then go to the Hololens analysis

5) Optional: Data Parsing:
6) Data visualization
7) K-neighbors
8) TPOT Analysis
9) Autoencoder Classifier

### Prerequisites

To use our codes, you will be requiring specific python packages. These include

```
json
numpy
pandas
glob
seaborn
sklearn
matplotlib
mpl_toolkits
sklearn
keras
PIL
tpot
```
<<<<<<< HEAD
## How to create a csv file with the features extracted from the json file

from /.Kate library/Data_Parsing import simple_features_generator

relative_path = '../Hololens_data/'
game_files=[
           relative_path + '5.json',
           relative_path + '6.json',
           relative_path + '7.json',
           relative_path + '8.json',
           relative_path + '9.json',
           relative_path + '10.json']

X,y = simple_features_generator(game_files, 1)

## Authors

* **Jose MEenoci Neto** - [netomenoci](https://github.com/netomenoci)
=======
## Running the tests

You are now ready to run our programs either in a Jupyter Notebook form or the standard .py form

## Authors

* **Neto Menoci** - [netomenoci](https://github.com/netomenoci)
>>>>>>> 5f0eb083e196a69ddd83f1e9701e998c053086bd
* **Gregoire Martin** - [GregoireMartin](https://github.com/GregoireMartin)
* **Niraj Srinivas** - [nirajsrinivas](https://github.com/nirajsrinivas)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details

## Acknowledgments

We would like to express our sincere thanks to Dr. Guillaume Dumas who has been very helpful to us, accompanying and help us during this semester, particularly in our interactions with Actimage.

We would also like to thank Dr. Letort-Le Chevalier and Dr. Myriam Tani who supervised our project and have accompanied us in several of our decisions.  Finally we would like to thank Actimage for their collaboration and database of the Hololens


