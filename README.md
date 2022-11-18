# Learning Unsupervised Hierarchies of Audio Concepts

This repository contains the Python (Tensorflow) code to reproduce the results of our accepted paper ["Learning Unsupervised Hierarchies of Audio Concepts"](https://arxiv.org/pdf/2207.11231.pdf) by D. Afchar, R. Hennequin and V. Guigue, that will be presented at [ISMIR 2022](https://ismir2022.ismir.net/) (December 4-8, 2022).

## Note

**:hammer: This is a work in progress :wrench:**

We provide the algorithmic part of our experiments, which can be already used by interested readers to check implementation details prior to the conference. The code may need a bit of cleaning, which will be done soon enough.

Supposedly, we may be able to upload most training data, but due to copyright issues, we still need to figure out how to properly anonymise everything.

## Result demo

You can access results demos at [research.deezer.com/concept_hierarchy](http://research.deezer.com/concept_hierarchy/), which includes useful interactive visualisations of the obtained graphs, as well as additional figures to help better interpret our (dense) tables of results.

## Usage

If you want to run scripts for this repo you have multiple options :

* Using docker : run ```make docker SCRIPT=1_CAV_train.py``` 
* using poetry : run `````poetry install````` and then execute python code via \
```poetry run python concept_hierarchy/script.py```
* using pip : run ```pip install -r requirements.txt``` and then execute using python

# Usage for Mac OS ARM
See https://developer.apple.com/metal/tensorflow-plugin/

## References

Contact: [research@deezer.com](mailto:research@deezer.com)

Consider citing our paper if you use our method in your own work:

```BibTeX
@inproceedings{afchar2022learning,
  title={Learning Unsupervised Hierarchies of Audio Concepts},
  author={Afchar, Darius and Hennequin, Romain and Guigue, Vincent},
  booktitle={International Society of Music Information Retrieval Conference (ISMIR)},
  year={2022}
}
```
