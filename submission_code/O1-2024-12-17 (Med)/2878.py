class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # ops[i] will track how many "decrements" start at index i.
        # We'll use it with a running cumulative sum to know how many
        # times each index has already been decremented.
        ops = [0]*(n+1)
        
        current_decrement = 0  # This will hold the cumulative sum of decrements as we iterate.
        
        for i in range(n):
            current_decrement += ops[i]  # Add any decrements that start at index i.
            # Determine how much we still need to decrement this position to get it to zero.
            needed = nums[i] - current_decrement
            
            # If needed is negative, we've overly decremented this position, so it's impossible.
            if needed < 0:
                return False
            
            # If we still need to decrement, we must apply it to the subarray [i, i+k-1].
            if needed > 0:
                if i + k > n:  # We can't form a subarray of length k starting at i.
                    return False
                current_decrement += needed         # Now each future index is further decremented by 'needed'.
                ops[i + k] -= needed               # Beyond i+k, the effect of this operation stops.
        
        return True