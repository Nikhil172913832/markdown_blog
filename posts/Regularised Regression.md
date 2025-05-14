#ml 

1. Ridge Regression : α∑<sup>n</sup><sub>i = 1</sub>  = 1 θ<sup>2</sup><sub>i</sub> is the regularisation term added to the cost function. Here even if the alpha is 1 then all weights end up being very close to zero.
2. Lasso Regression : Least Absolute Shrinkage and Selection Operator Regression adds α∑<sup>n</sup><sub>i = 1</sub>  |θ<sub>i</sub>| if alpha is one here it would make the weights go to zero.
3. Elastic Net: It is a combination of both rα∑<sup>n</sup><sub>i = 1</sub>  |θ<sub>i</sub>| + (1-r)/2*α∑<sup>n</sup><sub>i = 1</sub>  = 1 θ<sup>2</sup><sub>i</sub>. 
4. Use ridge by default instead of linear as regularisation is important.
5. If only a few features are actually useful then use lasso or elastic net as it would make weights of useless features go to zero , preferably use elastic as lasso behaves erratically when the number of features is > the number of instances or when several features are strongly co related.

