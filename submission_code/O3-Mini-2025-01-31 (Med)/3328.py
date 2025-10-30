import math

class Solution:
    def minOperations(self, k: int) -> int:
        # Special case: initial array [1] already has sum 1 which is >= k if k==1.
        if k == 1:
            return 0
        
        min_ops = float('inf')
        # We want to choose a target value "x" to which we increment the initial element.
        # Then we will duplicate that element until the array has enough copies.
        # The cost is: increments = (x - 1), duplications = (ceil(k / x) - 1)
        # Total cost = (x - 1) + (ceil(k/x) - 1) = x + ceil(k/x) - 2
        # We iterate over possible x values from 1 up to k (k is at most 10^5 so this is efficient)
        for x in range(1, k + 1):
            dup_ops = math.ceil(k / x) - 1
            total_ops = (x - 1) + dup_ops  # which is also equal to x + math.ceil(k/x) - 2
            if total_ops < min_ops:
                min_ops = total_ops
        return min_ops

# Example Usage:
# sol = Solution()
# print(sol.minOperations(11))  # Output: 5
# print(sol.minOperations(1))   # Output: 0