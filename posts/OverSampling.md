#ml 

It is a technique used to balance imbalanced datasets by duplicating existing data points of minority class or synthetically generating new data samples based on data patterns.
1. Random OverSampling: Randomly select data points of minority class and duplicate them without alteration. 
2. Smoothed bootstrap OverSampling: In this instead of just duplicating samples you make synthetic samples by interpolation of feature vectors of the sample class.
Problem with random oversampling is that its just mere duplication of data so the model might become over tailored to specific nuances of the minority class and would end up capturing a lot of noise from the training data.
3. 