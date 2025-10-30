from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # In the worst case, if we remove until the remaining array is empty,
        # we would have performed at most (n+2)//3 operations.
        max_ops = (n + 2) // 3
        
        for ops in range(max_ops + 1):
            # Calculate the starting index after performing ops removals.
            start_index = ops * 3
            # If start_index is within the list, take the remaining part,
            # otherwise it's an empty list which is distinct.
            remaining = nums[start_index:] if start_index < n else []
            # Check if remaining has distinct elements.
            if len(remaining) == len(set(remaining)):
                return ops
        
        # This return is just for a complete control flow; given the problem,
        # it will always be reached before this point.
        return max_ops