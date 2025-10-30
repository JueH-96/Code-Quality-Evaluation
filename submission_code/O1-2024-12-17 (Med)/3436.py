class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # We will keep track of all distinct bitwise-OR values of subarrays 
        # that end at the current index. This set is updated as we iterate.
        # It is known that the number of distinct OR values for subarrays 
        # ending at a particular index is at most ~32 (due to limited 32-bit spread).
        
        current_or_set = set()
        best_diff = abs(nums[0] - k)  # Initialize with the first element's difference
        
        for x in nums:
            new_or_set = {x}  # A subarray of length 1 includes only this element
            # Extend all previously possible OR values by ORing with current x
            for val in current_or_set:
                new_or_set.add(val | x)
            
            # Update the rolling set for next iteration
            current_or_set = new_or_set
            
            # Check each OR result to update the minimum difference
            for val in current_or_set:
                diff = abs(val - k)
                if diff < best_diff:
                    best_diff = diff
                    if best_diff == 0:
                        return 0  # Early exit if we cannot do better than 0
        
        return best_diff