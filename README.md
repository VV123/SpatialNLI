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

  
## How to use
  
