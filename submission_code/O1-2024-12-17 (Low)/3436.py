class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # We will maintain, for each ending index r, a set of distinct bitwise OR
        # values of subarrays that end exactly at r.
        # Reasoning:
        # 1) If we have all OR-values of subarrays ending at r-1 in a set,
        #    then appending nums[r] to each of those subarrays leads to new OR-values.
        # 2) Subarray of length 1 ending at r has OR-value nums[r].
        #
        # Because each new OR can only set bits, the number of distinct OR-values
        # does not explode beyond a certain bound (roughly up to ~32 for 32-bit
        # integers). Hence, this approach is O(N * 32) in the worst case.
        
        curr_or_values = set()
        min_diff = abs(k - nums[0])
        
        for num in nums:
            new_or_values = set()
            # Build new OR-values by extending previous subarrays plus starting fresh
            for val in curr_or_values:
                new_or_values.add(val | num)
            new_or_values.add(num)
            
            # Update our running minimum difference
            for val in new_or_values:
                diff = abs(k - val)
                if diff < min_diff:
                    min_diff = diff
                    if min_diff == 0:
                        return 0  # Early exit if we hit perfect match
            
            curr_or_values = new_or_values
        
        return min_diff