% Demo of psNMF
addpath(genpath('./'))

Ndims = [50,49]; D = 5;
constr = {'sparse','ard'};
X = generateTensorData(Ndims,D,[3,3]);
X = addTensorNoise(X,10); % SNR = 10 dB
VB_CP_ALS(X,D,constr,'maxiter',100, 'inference','variational');