class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate remainder when purchaseAmount is divided by 10
        remainder = purchaseAmount % 10
        # Calculate the lower multiple of 10
        lower = purchaseAmount - remainder
        # Calculate the higher multiple of 10
        higher = lower + 10
        
        # Determine the rounded amount per criteria:
        # if the difference to lower equals the difference to higher, take the higher multiple.
        if remainder < 5:
            roundedAmount = lower if (5 - remainder) > remainder else higher
        else:
            roundedAmount = higher
        
        # Alternatively, the rounding procedure can be implemented in a straight-forward manner:
        # Calculate distances:
        # distance_lower = purchaseAmount - lower
        # distance_higher = higher - purchaseAmount
        # if distance_higher < distance_lower, choose higher, else if same, choose higher.
        # But the above logic using remainder covers it adequately.
        
        # After calculating the payment amount, subtract it from the starting balance of 100 dollars
        return 100 - roundedAmount