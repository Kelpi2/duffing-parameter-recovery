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
 
 08-08-26
 Smaller bugs: network file had a colon which is illegal on windows - replaced with _,when defining path for figures it shared them with data - changed var name
SNRLoop collected R2 and RMSE per level which led to inconsitent saving in network data file, fixed by saving the runs in seperate data files

13-08-26
Split recoverparams so the snr loop and the actual parameter recovery are two different functions
Decided on generating 2000 data points for each trajectory and passing it to AR and linear reg but keeping mlp at 604. This is due to the decimation whic would result in only 24 samples with 604 points, mlp wins eitherway.

16-08-26
Edited some pre-existing functions to work in experiments.py
Decimation wasnt working in experiments due to the timesteps being off,fixed by multiply 0.063 by 25

dataPrep was accidently re-assigning a new value to Y, so once it moved on to the next SNR level it got a "index 2 is out of bounds for axis 1 with size 2" error. Fixed by renaming the parameter to Truthlabel
NanCounter wasnt working, it was checking for np.nan==np.nan which is against numpy rules, changed to np.isnan(alpha)

Did a decimation sweep for linear reg as the noise multiplication due to passing variables through FDV() twice diminished results. Found that the values for optimal dec vary a lot per snr level, for example at snr 1 at dec 1(no decimation) recovered param are 104,0.03 as opposed to the true values of 1.25,0.3, at a dec of 10 it is 1.13,0.079. However the opposite is seen on the clean dataset where a dec of 1 gives 1.24,0.29 and 1.06,2.23 at dec 10. To fix this I ran the sweep and find the optimal dec for each snr level and got these results [1, 2, 6, 8, 9, 10]