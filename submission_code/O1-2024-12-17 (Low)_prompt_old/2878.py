class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # This array will track how many decrement operations
        # stop affecting indices at each point.
        diff_end = [0] * (n + 1)
        
        current_decrement = 0  # Track how many times we've subtracted so far as we move along
        
        for i in range(n):
            # Add back any decrements that have ended at this index
            current_decrement += diff_end[i]
            
            # Effective value at nums[i] after all decrements so far
            needed = nums[i] - current_decrement
            
            # If we already over-subtracted (needed < 0) it's not fixable
            if needed < 0:
                return False
            
            # If we need to subtract more from this position forward
            if needed > 0:
                # If there's not enough space left to apply k-length subarray
                if i + k > n:
                    return False
                
                # We apply 'needed' sub-operations from i to i+k-1
                current_decrement += needed
                
                # Mark that these 'needed' sub-operations stop after index (i+k-1)
                diff_end[i + k] -= needed
        
        return True