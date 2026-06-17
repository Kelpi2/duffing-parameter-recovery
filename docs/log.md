Theres no previous bug history as i started it today to keep a record, previous changes can be found in commit history
10-06-2026
Caught a failed lr reset attempt which caused the decayed lr to leak into subsequent SNR levels leading to a tiny lr, fixed by changing lr = startlr
