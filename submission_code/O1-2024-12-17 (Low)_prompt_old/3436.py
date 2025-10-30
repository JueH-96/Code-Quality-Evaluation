class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # We will keep track of all possible bitwise OR values that end at each index.
        # For position i, we build new OR values from all values we had at position i-1 plus nums[i].
        # It is known that, for a sequence of OR accumulations, the set of distinct values
        # increases slowly (bounded by roughly the number of bits).
        
        best_diff = abs(k - nums[0])
        prev_or_values = set()  # possible OR values for subarrays ending at i-1
        
        for x in nums:
            new_or_values = {x}
            # Extend each previously possible OR with current num
            for val in prev_or_values:
                new_or_values.add(val | x)
                
            # Update our best difference using these new OR values
            for val in new_or_values:
                best_diff = min(best_diff, abs(k - val))
            
            # Prepare for the next iteration
            prev_or_values = new_or_values
        
        return best_diff