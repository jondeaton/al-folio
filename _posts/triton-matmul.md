---
layout: post
title: Triton Tutorial: Matrix Multiply Difficulties
date: 2023-09-04
description: Discrepancies with Triton matrix multiply tutorial
tags: programming
featured: true
---

working through the Triton [matmul tutorial](
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py
)


Having several issues
1. My implementation results in a significantly wrong result, likely
   because I have a bug in my code.

2. When I copy their implementation, I end up with a very slight difference in
   the numerical result, and doesn't test the "unit test". There is an error of
   ~0.03

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

This can only happen for two reasons. (1) there's a bug in the Triton tutorail,
or (2) something I've been worried about for along time: my CUDA/cuBLAS has been
installed incorrectly.

3. When I run the benchmarking, even with the tutorial's implementation, I get
   significantly worse performance than cuBLAS. The tutorial shows that we
    should be able to get essentially equivalent performance.

![Mine](triton_matmul/matmul_perf_mine.png)
![Theirs](assets/img/triton_matmul/matmul_perf_theirs.png)

matmul-performance:
         M     cuBLAS     Triton
0    256.0   5.140079   2.641249
1    384.0   9.857783   6.627236
2    512.0  19.284157  12.501652
3    640.0  25.600001  13.546093
4    768.0  28.142696  14.459424
5    896.0  39.025776  19.919228
6   1024.0  51.582524  26.214401
7   1152.0  35.941882  17.219587
8   1280.0  44.521738  21.113403
9   1408.0  54.263400  26.088954
10  1536.0  48.918450  20.987066
11  1664.0  51.025992  24.680967
12  1792.0  54.092580  28.079913
13  1920.0  51.654367  24.399107
14  2048.0  58.629563  27.688030
15  2176.0  44.692837  25.216617
16  2304.0  50.033505  27.640501
17  2432.0  54.658489  26.049574
18  2560.0  59.539279  27.959045
19  2688.0  50.442892  26.907350
20  2816.0  54.970838  25.929969
21  2944.0  57.860570  27.860328
22  3072.0  51.732005  27.139873
23  3200.0  54.421769  26.492121
24  3328.0  57.686777  28.084232
25  3456.0  54.547747  27.994512
26  3584.0  56.167501  27.581409
27  3712.0  52.327104  27.219638
28  3840.0  54.337456  26.837324
29  3968.0  52.038224  26.998499
30  4096.0  55.052392  27.017641


So what is the reason for these issues?

