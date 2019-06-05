# Spatial-Comprehension
Understanding Spatial Semantics based on context. (e.g., understand the type of a Point-of-Interest)

## Challenge
 | Context | Spatial Semantics |
 |---------|:-------------------:|
 |  | The meaning of Spatial phrase `Mississippi` |
 | How many people live in `Mississippi` ?| State |
 | How many states does the `Mississippi` run through ?| River |
 |  |  The meaning of Spatial phrase `Over`  |
 | How many people walked `over` the bridge? | On |
 | How many birds flew `over` the bridge?    | Above|
 | | The meaning of spatial phrase `at the back of`  |
 | How many trees are `at the back of` the building? | Exterior |
 | How many rooms are `at the back of` the building? | Interior |
  
   
## Approach
  Using a bi-directional attention flow to resolve ambiguity by understanding the spatial context.
  ![Model](model.jpg)

## Usage

To train a new model

```python word_classifier.py --mode train```

To use trained model for evaluating test sets

```python word_classifier.py --mode infer```

To use trained model to infer a question

For example, "How many rivers are found in `colorado` ?", to infer whether `coloado` refers to a `city`, `state`, or `river`.

    ls = ['how many rivers are found in <f0> colorado <eof>\tcity', 'how many rivers are found in <f0> colorado <eof>\tstate', 'how many rivers are found in <f0> colorado <eof>\triver'] 
    tf_model = TF()
    g = glove.Glove()
    flag, prob = tf_model.infer(ls, g)
    ---------------------------------
    OUTPUT: flag = 'state'
            prob = [.1, .9, .1]
    
    



 
 

## Performance
 
  
  |Data Split|         | Train | Test|
  |----------|---------|-------|-----|
  |Geoquery  | Acc     |98.3%  |98.1%|
  |Restaurant| Acc     |100%   |100% |
  

<!---Training with multiple datasets using a single model
|              |Acc<sub>qm</sub>|Acc<sub>qm</sub>|
|--------------|----------------|----------------|
| Model        |Geoquery        |Restaurant      |
| Separate     | 90.4%          |100%            |
| Shared       |**90.7%**       |**100%**        |
--->

  
## Miscellaneous

  Make sure the size of the test set is divisible by batch_size while evaluation.


  
