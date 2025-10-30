class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # This will keep track of the number of decrements we need to apply
        # to the current position due to previous operations
        decrement_effect = [0] * (n + 1)
        
        current_decrement = 0
        
        for i in range(n):
            # Apply the decrement effect from previous operations
            current_decrement += decrement_effect[i]
            
            # Calculate the effective value at nums[i] after applying all decrements
            effective_value = nums[i] - current_decrement
            
            # If the effective value is negative, it means we have over-decremented
            if effective_value < 0:
                return False
            
            # If the effective value is positive, we need to perform more operations
            if effective_value > 0:
                # If we are too close to the end and cannot form a subarray of size k
                if i + k > n:
                    return False
                
                # We need to perform 'effective_value' operations starting from index i
                current_decrement += effective_value
                # Mark the end of the effect of this operation
                decrement_effect[i + k] -= effective_value
        
        return True