# Submission docs
Our team (Erik, Jan, Ali, Alex, Augustin) pursued two separate approaches:
1. Creating a custom potential, that guides the VL-VH angle towards an empirical angle distributione extracted from PDBase
2. Optimizing which model output metrics should be used for ranking the predicted structures

## Antibody Angle Potential
Following [this paper](https://www.researchgate.net/publication/236939459_ABangle_Characterising_the_VH-VL_orientation_in_antibodies) we define a potential based on the angle between VL and HL. 
We extract X Antibody VL-HL pairs from PDBase and calculate their empirical angle histogram.
[IMG LINK]

## Ranking Metric Optimization
