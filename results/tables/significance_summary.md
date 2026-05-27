# Significance Test Summary

ML vs baseline pairs tested : 32  (8 ML configs x 4 baselines)
Bonferroni threshold        : p < 0.00156  (alpha=0.05 / 32)
Significant after Bonferroni: 0 / 32
Significant after BH FDR    : 0 / 32

## Per-comparison results  (sorted by raw p-value)

ML model           variant  baseline          dReturn   dSharpe       DM     p_raw    p_bonf      p_BH   sig*
-------------------------------------------------------------------------------------------------------------
logistic           regime   equal_weight      +0.0517    +0.091   +1.809    0.0709    1.0000    0.4311     no
linear             regime   equal_weight      +0.0515    +0.044   +1.773    0.0767    1.0000    0.4311     no
pca                base     spy_bh            -0.0896    -0.348   -1.728    0.0844    1.0000    0.4311     no
linear             base     equal_weight      +0.0522    +0.008   +1.687    0.0920    1.0000    0.4311     no
lstm               base     spy_bh            -0.1087    -0.745   -1.625    0.1046    1.0000    0.4311     no
pca                base     momentum_12_1     -0.0573    -0.496   -1.574    0.1161    1.0000    0.4311     no
linear             regime   sixty_forty       +0.0727    +0.371   +1.519    0.1293    1.0000    0.4311     no
linear             base     sixty_forty       +0.0734    +0.336   +1.503    0.1334    1.0000    0.4311     no
logistic           regime   sixty_forty       +0.0729    +0.419   +1.501    0.1339    1.0000    0.4311     no
logistic           base     equal_weight      +0.0434    +0.017   +1.498    0.1347    1.0000    0.4311     no
lstm               regime   spy_bh            -0.0889    -0.627   -1.406    0.1603    1.0000    0.4659     no
pca                regime   spy_bh            -0.0623    -0.233   -1.318    0.1879    1.0000    0.4659     no
logistic           base     sixty_forty       +0.0646    +0.344   +1.314    0.1893    1.0000    0.4659     no
lstm               base     momentum_12_1     -0.0764    -0.894   -1.207    0.2279    1.0000    0.5209     no
pca                base     equal_weight      -0.0309    -0.533   -1.139    0.2553    1.0000    0.5446     no
lstm               regime   momentum_12_1     -0.0566    -0.776   -0.939    0.3482    1.0000    0.6963     no
lstm               base     equal_weight      -0.0500    -0.930   -0.861    0.3895    1.0000    0.7331     no
pca                regime   momentum_12_1     -0.0300    -0.382   -0.797    0.4259    1.0000    0.7572     no
linear             base     momentum_12_1     +0.0257    +0.045   +0.642    0.5210    1.0000    0.8078     no
linear             regime   momentum_12_1     +0.0250    +0.080   +0.641    0.5218    1.0000    0.8078     no
logistic           regime   momentum_12_1     +0.0252    +0.128   +0.628    0.5301    1.0000    0.8078     no
lstm               regime   equal_weight      -0.0301    -0.812   -0.532    0.5950    1.0000    0.8655     no
lstm               base     sixty_forty       -0.0287    -0.603   -0.463    0.6438    1.0000    0.8930     no
logistic           base     momentum_12_1     +0.0169    +0.053   +0.427    0.6698    1.0000    0.8930     no
pca                regime   sixty_forty       +0.0177    -0.091   +0.368    0.7129    1.0000    0.9110     no
logistic           base     spy_bh            -0.0154    +0.202   -0.311    0.7557    1.0000    0.9110     no
pca                base     sixty_forty       -0.0096    -0.205   -0.219    0.8271    1.0000    0.9110     no
linear             regime   spy_bh            -0.0073    +0.229   -0.151    0.8800    1.0000    0.9110     no
lstm               regime   sixty_forty       -0.0089    -0.485   -0.146    0.8843    1.0000    0.9110     no
logistic           regime   spy_bh            -0.0071    +0.277   -0.142    0.8873    1.0000    0.9110     no
linear             base     spy_bh            -0.0066    +0.194   -0.139    0.8892    1.0000    0.9110     no
pca                regime   equal_weight      -0.0035    -0.418   -0.112    0.9110    1.0000    0.9110     no

\* Bonferroni-corrected at α=0.05

## Bottom line

After Bonferroni correction across 32 comparisons, **0 of 32 ML-vs-baseline pairs are significant at p<0.05**. We cannot reject the null hypothesis that the ML models produce the same expected returns as the passive baselines.