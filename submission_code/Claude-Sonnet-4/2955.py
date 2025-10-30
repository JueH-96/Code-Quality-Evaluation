class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Find the two nearest multiples of 10
        lower_multiple = (purchaseAmount // 10) * 10
        upper_multiple = lower_multiple + 10
        
        # Calculate distances
        distance_to_lower = abs(purchaseAmount - lower_multiple)
        distance_to_upper = abs(purchaseAmount - upper_multiple)
        
        # Choose the nearest multiple, with preference for larger if equal
        if distance_to_lower < distance_to_upper:
            rounded_amount = lower_multiple
        else:  # distance_to_upper <= distance_to_lower
            rounded_amount = upper_multiple
            
        return 100 - rounded_amount