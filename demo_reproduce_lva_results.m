%% Add relevant paths

% The psNMF toolbox
addpath(genpath('./'));

% Path to [1] explicit sparse NMF (PH-sNMF), specificly function "nmfsc".
addpath('../../thirdparty-matlab/nmfpack/code/')
% For load faces and PH-NMF (nmfsc)

% Path to [2] implicit sparse NMF (YA-sNMF), specificly function "sparsenmfnnls".
addpath('../../thirdparty-matlab/nmfv1_4/')

% Path to [1] NMF, specificly function "cpNonNeg".
addpath('../../thirdparty-matlab/cpNonNeg-master/')

% Get the MNIST data and load function from [4]
addpath('../../thirdparty-matlab/mnist-tools/')


%% References
% [1] Hoyer, P.O.: Non-negative matrix factorization with sparseness
%      constraints. Journal of machine learning research 5(Nov) (2004) 1457-1469 
%
% [2] Li, Y., Ngom, A.: The non-negative matrix factorization toolbox for
%      biological data mining. Source code for biology and medicine 8(1) (2013)
%
% [3] Nielsen, S.F.V., Mørup, M.: Non-negative tensor factorization with
%      missing data for the modeling of gene expressions in the human brain. In:
%      Machine Learning for Signal Processing (MLSP), 2014 IEEE International
%      Workshop on, IEEE (2014)   
%
% [4] LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.: Gradient-based
%      learning applied to document recognition. Proceedings of the IEEE 86(11)
%      (1998) 2278-2324  

%%

%% Run all LVA experiments (once)
% Ensure that the path to the datasets have been modified in "LVA_experiments"
LVA_experiments(1,1,1,'./run-LVA-ICA/results','test-run')
% or 
% LVA_experiments

%% Visualize the results
% Given that they are saved to './run-LVA-ICA/results'
LVA_Vis