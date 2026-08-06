Theres no previous bug history as I started it today to keep a record, previous changes can be found in commit history
10-06-2026
Caught a failed lr reset attempt which caused the decayed lr to leak into subsequent SNR levels leading to a tiny lr, fixed by changing lr = startlr


25-06-26 Day 6
Caught a minor bug in loss curve graph. Loss was being calculated with un-normalised values of states and normalised values of preds

28-07-26 Day 9
Ar_model is returning nan for alpha,freq results and large values for gamma
Turned out to be accurate results as a2 was coming out as positive so "np.sqrt(-a2)" returned nan

29-07-26
rewrote fit from a fixed 2nd order fitter to a variable one

30-07-26
Predict function is returning a large numpy array per SNR level, however it should only return the next displacement value
Fix: Entire prediction array was being printed out except only the newly generated values

Added a decimation value to recoverParam as the small timestep value adequate for rk4 causes a2 and the arcos argumen to sit at their limits(-1 and 1) almost regardless of gamma and alpha, leaving too little signal for the fir to survive noise.

02-08-26
TrainSet[traj] = states was writing a (604,2) array inyo a (604,) row fixed by slicing the states array:states[:604, 0]

loss() was returning an array so epochloss accumulated an array and the yaxis[epoch] assignment failed. Fixed by taking the mean over both axis

Model was overfitting with the largest gap being 0.2 on gamma. 
Improved by generating more rows, from 1000 to 5000 for 4000 training points

05-08-26
Added evaluation for each epoch to further reduce overfitting
Lists weren't properly copied so minWeights wasnt calculated properly, changed to .copy() as the fix

06-08-26
At snr 1 the model is still trying to learn noise rather than the parameters. Minimum is at epoch 9 and 
Fixed by changing how prepareData() works, split then add noise, keeping X_train clean and adding noise to training data in trainLoop() every epoch

X_train was being reset to its original order every epoch unlike Y_train which stayed in its shuffled order causing a mismatch in trajectories to parameters causing the network to regress.
Fixed by creating new variables X_shuff,Y_shuff rather than manipulating original training sets
 