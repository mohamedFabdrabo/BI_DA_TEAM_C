README - Exercise Sheet 8 - Group C

Group Members:
- Federico
- Elisa
- Mohamed

Contribution Log:

All group members worked on Exercise Sheet individually at first. 
We then discussed our approaches and solutions together, and agreed on a final submission based on the most accurate and well-documented results.

The individual contributions are as follows:

Task 1:
- Contributor: Elisa
- Details: - Simulate particles from the SDE up to a final time T = 2.0 using the Euler–Maruyama method.
		   -  Compare the histogram of the particle distribution at times t ∈ {0.5, 1.0, 2.0} to the analytical solution.
		   -  Briefly discuss whether the empirical distribution matches the analytical solution at each time point.
		   
Task 2:
- Contributors: Federico
- Details: - Well-specified case: The rank histogram is relatively flat, indicating that the ensemble forecast accurately represents the distribution of the true process. 
				There is no significant skewness or shape bias, suggesting that the forecast is both unbiased and well-calibrated.
		   -  Misspecified case: The rank histogram exhibits a pronounced U-shape, with a high frequency of ranks at 0 and 100. This suggests that the ensemble forecast consistently underestimates the variability of the true process.
				The observation frequently falls outside the ensemble spread, revealing underdispersion and significant model bias due to incorrect parameter choices.


Task 3:
- Contributor: Mohamed
- Details: 
	- Show that (4.45) and (4.44) are indeed equivalent using the definition of the continuous ranked probability score (4.43) (In the Book).
	- Proving that the Score function is proper
	- Using the forecasts from Problem 2, compute the empirical CRPS for the well-specified and misspecified ensembles, 
		and discuss how the scores corroborate the conclusions drawn from the rank histograms.


Notes:
- All solutions were discussed and validated as a group before finalizing this submission.
- All Python code was written using Python 3.x and contains comments explaining the functionality.
- This submission follows all formatting and content requirements as outlined in the official assignment submission guidelines.