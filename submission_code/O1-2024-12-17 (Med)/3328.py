class Solution:
    def minOperations(self, k: int) -> int:
        # If k is already 1, no operations are needed
        if k == 1:
            return 0
        
        # We'll try building a single "big" number x by (x - 1) increments,
        # then duplicate it enough times so that the sum >= k.
        # For each x, the cost is:
        #   cost(x) = (x - 1) + (ceil(k / x) - 1)
        # We search over x in [1..k] and take the minimum.
        
        import math
        
        min_cost = float('inf')
        for x in range(1, k + 1):
            # increments to get a single element of value x
            increments = x - 1
            # number of duplicates needed to ensure sum >= k
            # each duplicate adds one more x to the array, so we need
            # at least ceil(k / x) total copies. We start with 1 copy,
            # so duplicates = total_needed - 1
            duplicates = (k + x - 1) // x - 1
            cost = increments + duplicates
            min_cost = min(min_cost, cost)
        
        return min_cost