
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
   because I have a bug

2. When I copy their implementation, I end up with a very slight difference in
   the numerical result, and doesn't test the "unit test". There is an error of
   ~0.03

3. When I run the benchmarking, even with the tutorial's implementation, I get
   significantly worse performance than cuBLAS. The tutorial shows that we
    should be able to get essentially equivalent performance.

![Mine](triton_matmul/matmul_perf_mine.png)
![Theirs](assets/img/triton_matmul/matmul_perf_theirs.png)
