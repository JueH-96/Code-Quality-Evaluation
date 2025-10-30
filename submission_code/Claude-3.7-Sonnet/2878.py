class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # difference array to track decrement operations
        decrement = [0] * (n + 1)
        
        current_decrement = 0
        for i in range(n):
            # Apply the current pending decrement at this position
            current_decrement += decrement[i]
            
            # After applying decrements, calculate what remains
            remaining = nums[i] - current_decrement
            
            # If we've decremented too much (element would become negative)
            if remaining < 0:
                return False
            
            # If we need to decrement this element more to make it zero
            if remaining > 0:
                # Check if we can apply an operation (need k elements from here)
                if i + k > n:
                    return False
                
                # Apply a decrement operation starting at i, affecting k elements
                decrement[i] += remaining
                decrement[i + k] -= remaining  # Cancel effect after k positions
                current_decrement += remaining
        
        return True