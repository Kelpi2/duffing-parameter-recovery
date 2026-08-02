Theres no previous bug history as i started it today to keep a record, previous changes can be found in commit history
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
trainSet[traj] = states was writing a (604,2) array inyo a (604,) row fixed by slicing the states array:states[:604, 0]

loss() was returning an array so epochloss accumulated an array and the yaxis[epoch] assignment failed. Fixed by taking the mean over both axis


