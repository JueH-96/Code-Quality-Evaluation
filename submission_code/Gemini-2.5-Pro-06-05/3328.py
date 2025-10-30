class Solution:
    def minOperations(self, k: int) -> int:
        
        if k == 1:
            return 0
        
        # The strategy is to form an array of `m` identical elements, each with value `a`.
        # The number of operations for this is (a - 1) increments and (m - 1) duplications.
        # Total ops = a + m - 2.
        # We need the sum `m * a` to be at least `k`.
        #
        # For a fixed `m`, we should choose the smallest `a` that satisfies `m * a >= k`.
        # This is `a = ceil(k / m)`.
        #
        # We can iterate through possible values of `m` to find the minimum total operations.

        # The cost for m=1 is k-1. This is our initial best answer.
        min_ops = k - 1
        
        # Iterate through number of elements `m` from 2 upwards. We don't need to check
        # beyond m=k, but our optimization will stop the loop much earlier.
        for m in range(2, k + 1):
            
            # Optimization: If the number of duplications alone (m-1) is already
            # equal or greater than the best answer found so far, we can stop.
            # Any further increase in `m` will only lead to more operations, since
            # the number of increments (a-1) is non-negative.
            if m - 1 >= min_ops:
                break
            
            # For a given `m`, calculate the smallest `a` such that `m * a >= k`.
            # a = ceil(k / m) which is (k + m - 1) // m in integer arithmetic.
            a = (k + m - 1) // m
            
            # Total operations for this configuration.
            current_ops = (a - 1) + (m - 1)
            
            # Update the minimum operations found.
            min_ops = min(min_ops, current_ops)
            
        return min_ops