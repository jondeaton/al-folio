---
layout: post
title: GPU Matrix Multiplication Inconsistencies
date: 2023-10-14
description:
tags: programming
featured: true
---

While working through the [Triton Matrix Multplication Tutorial]() I got an
error on the numerical results of the Triton vs Torch matrix multiply.

```
Mismatched elements: 706 / 262144 (0.269%)
Max absolute difference: 0.03125
Max relative difference: 0.05948
 x: array([[ 13.32  ,  17.27  ,  24.02  , ..., -22.66  ,  15.664 ,  -5.043 ],
       [ -9.32  ,  26.33  ,  -6.945 , ...,  31.94  ,  23.4   ,  18.11  ],
       [-11.96  ,  44.88  , -22.78  , ...,  20.98  , -56.5   ,  10.31  ],...
 y: array([[ 13.32  ,  17.27  ,  24.02  , ..., -22.66  ,  15.664 ,  -5.043 ],
       [ -9.32  ,  26.33  ,  -6.945 , ...,  31.94  ,  23.4   ,  18.11  ],
       [-11.96  ,  44.88  , -22.78  , ...,  20.98  , -56.5   ,  10.31  ],...
```

Not good.

Then I checked whether torch matmul even matches the numpy matrix multiply. To
my horror it doesn't even match.

torch gpu vs numpy
```
E           Mismatched elements: 627 / 262144 (0.239%)
E           Max absolute difference: 0.03125
E           Max relative difference: 0.08655
E            x: array([[ 13.32  ,  17.27  ,  24.02  , ..., -22.66  ,  15.664 ,  -5.043 ],
E                  [ -9.32  ,  26.33  ,  -6.945 , ...,  31.94  ,  23.4   ,  18.11  ],
E                  [-11.96  ,  44.88  , -22.78  , ...,  20.98  , -56.5   ,  10.31  ],...
E            y: array([[ 13.32  ,  17.27  ,  24.02  , ..., -22.66  ,  15.664 ,  -5.043 ],
E                  [ -9.32  ,  26.33  ,  -6.945 , ...,  31.94  ,  23.4   ,  18.11  ],
E                  [-11.96  ,  44.88  , -22.78  , ...,  20.98  , -56.5   ,  10.31  ],...
```

Digging into the problem deeper, I ran the same exact test using JAX with the
same float16 data types on the GPU, and again got a mismatch with the numpy
result.

JAX GPU results
```
E           Mismatched elements: 626 / 262144 (0.239%)
E           Max absolute difference: 0.03125
E           Max relative difference: 0.0204
E            x: array([[ 36.44  ,   1.076 , -30.48  , ...,  28.33  ,  23.45  ,  -9.94  ],
E                  [  9.61  ,  -9.69  ,  43.8   , ...,   7.184 ,  39.78  ,  -5.215 ],
E                  [ 37.3   ,  -3.887 ,  -0.764 , ...,  32.25  ,  20.62  ,   9.27  ],...
E            y: array([[ 36.44  ,   1.076 , -30.48  , ...,  28.33  ,  23.45  ,  -9.94  ],
E                  [  9.61  ,  -9.69  ,  43.8   , ...,   7.184 ,  39.78  ,  -5.215 ],
E                  [ 37.3   ,  -3.887 ,  -0.764 , ...,  32.25  ,  20.62  ,   9.27  ],...
``````
I tried using precision='high' and 'highest' but got the same exact results.

JAX CPU results
```
E           Mismatched elements: 455 / 262144 (0.174%)
E           Max absolute difference: 0.03125
E           Max relative difference: 0.02438
E            x: array([[ 36.44  ,   1.076 , -30.48  , ...,  28.33  ,  23.45  ,  -9.94  ],
E                  [  9.61  ,  -9.69  ,  43.8   , ...,   7.184 ,  39.78  ,  -5.215 ],
E                  [ 37.3   ,  -3.887 ,  -0.764 , ...,  32.25  ,  20.62  ,   9.27  ],...
E            y: array([[ 36.44  ,   1.076 , -30.48  , ...,  28.33  ,  23.45  ,  -9.94  ],
E                  [  9.61  ,  -9.69  ,  43.8   , ...,   7.184 ,  39.78  ,  -5.215 ],
E                  [ 37.3   ,  -3.887 ,  -0.764 , ...,  32.25  ,  20.62  ,   9.27  ],...
```

This indicates to me that the issue isn't so much

