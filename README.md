# Spatial-Comprehension
Understanding Spatial Point-of-Interest

## Challenge
  |       | Spatial POI Ambiguity                        |
  |-------|--------------------------------------|
  | State | How many people live in `Mississippi` ?| 
  | River | How many states does the `Mississippi` run through ?|
## Approach
  Using a bi-directional attention flow to resolve ambiguity by understanding the spatial context.
  ![Model](spatial-comprehension-model.pdf)
  
## Performance
  Geo880 Train Acc 100%  Test Acc 98%
  

<!---Training with multiple datasets using a single model
|              |Acc<sub>qm</sub>|Acc<sub>qm</sub>|
|--------------|----------------|----------------|
| Model        |Geoquery        |Restaurant      |
| Separate     | 90.4%          |100%            |
| Shared       |**90.7%**       |**100%**        |
--->

  
## How to use
  
