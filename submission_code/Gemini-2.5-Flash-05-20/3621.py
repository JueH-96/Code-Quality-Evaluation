from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Step 1: Check if it's impossible.
        # If any element in nums is initially less than k, it's impossible
        # because the operations can only decrease values or keep them the same.
        for x in nums:
            if x < k:
                return -1
        
        # Step 2: Collect all unique values from nums that are strictly greater than k.
        # These are the values that need to be reduced to k.
        unique_values_to_reduce = set()
        for x in nums:
            if x > k:
                unique_values_to_reduce.add(x)
        
        # Step 3: The minimum number of operations is the count of these unique values.
        # Each distinct value 'v' that is > k must be transformed. The operation
        # rule allows us to effectively 'eliminate' the current maximum value 'M'
        # from the set of values that are > k.
        # Specifically:
        # - If M is the only unique value > k (or all other unique values are <= k),
        #   we perform an operation using h=k, setting all M's to k. This costs 1 operation.
        # - If there is a second largest unique value M2 such that k <= M2 < M,
        #   we perform an operation using h=M2, setting all M's to M2. This costs 1 operation.
        #   This effectively reduces M to M2, and now M2 becomes the largest value (or
        #   part of the values to be processed). In both cases, one distinct value
        #   greater than k is "dealt with" per operation.
        # Thus, the total number of operations equals the number of distinct values
        # that were initially greater than k.
        
        return len(unique_values_to_reduce)