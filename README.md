# kogrobot

## Usage
Use with ```python3.6 main <dir>```
logs to ```results.txt```

## Method
1. Trained linear SVM to recognize the relevant colors (red, green, blue, yellow, white, other) of a pixel (works well, ~99% F-measure on test set)
2. For each image scan column-wise and use SVM to recognize pixel color.
3. If colors of column show all relevant color transitions (e.g. blue -> yellow -> white -> green (lawn)), column belongs to pylon.
4. Adjacent columns of pylons are grouped into one pylon.

## Evaluation
### Benefits
* Yields a high-precision (few false positives) pylon recognition algorithm
* Bottom border of pylon can be easily recognized: white -> green (lawn), and is detected accurately

### Problems
* Terribly slow (SVM gets called on each column)
* Top border of pylon is hard to recognize because transitions (e.g. other -> blue) may also be detected in pylon
