from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum possible number of distinct elements after performing
        the allowed operations.
        """
        nums.sort()
        
        count = 0
        # `last_val` stores the last distinct value that was assigned.
        # Initialize to a value lower than any possible outcome to handle the first element correctly.
        last_val = float('-inf')
        
        for num in nums:
            # We employ a greedy strategy. For each number, we try to assign it the
            # smallest possible new value that is still available. This leaves maximal
            # "space" on the number line for subsequent, larger numbers.
            # An available value must be:
            # 1. Greater than the last assigned value (i.e., at least `last_val + 1`).
            # 2. At least the minimum possible for the current number (i.e., `num - k`).
            # So, our target is the maximum of these two lower bounds.
            target = max(num - k, last_val + 1)
            
            # Now, we check if this smallest possible target is actually achievable.
            # It is achievable if it's within the modification range of the current number,
            # which is [num - k, num + k].
            if target <= num + k:
                # If the target is achievable, we have found a new distinct element.
                count += 1
                # Update the last value we placed to this new target.
                last_val = target
                
        return count