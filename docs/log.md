Theres no previous bug history as i started it today to keep a record, previous changes can be found in commit history
10-06-2026
Caught a failed lr reset attempt which caused the decayed lr to leak into subsequent SNR levels leading to a tiny lr, fixed by changing lr = startlr


25-06-26 Day 6
Caught a minor bug in loss curve graph. Loss was being calculated with un-normalised values of states and normalised values of preds

28-07-26 Day 9
Ar_model is returning nan for alpha,freq results and large values for gamma
Turned out to be accurate results as a2 was coming out as positive so "np.sqrt(-a2)" returned nan

rewrote fit from a fixed 2nd order fitter to a variable one