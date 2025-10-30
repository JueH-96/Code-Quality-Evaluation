from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Total number of distinct elements in the whole array
        total_distinct = len(set(nums))
        n = len(nums)
        complete_count = 0
        
        # For every possible starting index `i`
        for i in range(n):
            seen = set()                          # distinct elements in current window
            # Extend the window to the right, index `j`
            for j in range(i, n):
                seen.add(nums[j])
                
                # As soon as the window contains all distinct elements
                if len(seen) == total_distinct:
                    # Every extension of this window to the right
                    # will still contain all distinct elements.
                    # Number of such extensions = n - j.
                    complete_count += n - j
                    break
        
        return complete_count