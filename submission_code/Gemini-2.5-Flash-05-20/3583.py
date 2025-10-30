import math
import bisect
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Determine the maximum possible GCD value, which is the maximum value in nums.
        # This defines the upper bound for our calculations and array sizes.
        max_val = max(nums)
        
        # Step 1: Count frequencies of each number in `nums`.
        # `counts[x]` will store how many times `x` appears in `nums`.
        # The array is sized `max_val + 1` to accommodate values from 0 to max_val.
        counts = [0] * (max_val + 1)
        for x in nums:
            counts[x] += 1
            
        # Step 2: Calculate `cnt_multiples[g]`.
        # `cnt_multiples[g]` stores the total count of numbers in `nums` that are multiples of `g`.
        # This is computed by iterating through all possible GCD values `g` (from 1 to `max_val`)
        # and then summing up counts of all multiples of `g`.
        # Time complexity: O(M log M) due to harmonic series sum.
        cnt_multiples = [0] * (max_val + 1)
        for g in range(1, max_val + 1):
            for multiple in range(g, max_val + 1, g):
                cnt_multiples[g] += counts[multiple]
        
        # Step 3: Calculate `exact_gcd_count[g]`.
        # `exact_gcd_count[g]` stores the precise number of pairs (nums[i], nums[j]) with `i < j`
        # such that `gcd(nums[i], nums[j]) == g`.
        # This is done using an inclusion-exclusion principle, iterating `g` downwards from `max_val` to 1.
        # For a given `g`, `pairs_with_gcd_multiple_of_g` initially counts all pairs whose elements
        # are *both* multiples of `g`. This count includes pairs whose GCD is `g`, `2g`, `3g`, etc.
        # We then subtract the `exact_gcd_count` for proper multiples of `g` (e.g., `2g, 3g, ...`),
        # which have already been calculated in previous iterations (since we iterate downwards).
        # Time complexity: O(M log M).
        exact_gcd_count = [0] * (max_val + 1)
        for g in range(max_val, 0, -1):
            # Calculate total pairs where both numbers are multiples of `g`.
            # If there are `C` numbers in `nums` that are multiples of `g`,
            # then there are `C * (C - 1) / 2` ways to choose 2 of them.
            num_elements_multiples_of_g = cnt_multiples[g]
            pairs_with_gcd_multiple_of_g = num_elements_multiples_of_g * (num_elements_multiples_of_g - 1) // 2
            
            # Subtract counts of pairs whose GCD is a multiple of `g` but strictly greater than `g`.
            # (e.g., if g=1, subtract exact_gcd_count for values like 2, 3, 4, etc.)
            for multiple_g_factor in range(2 * g, max_val + 1, g):
                pairs_with_gcd_multiple_of_g -= exact_gcd_count[multiple_g_factor]
            
            exact_gcd_count[g] = pairs_with_gcd_multiple_of_g
            
        # Step 4: Compute `cumulative_gcd_count[g]`.
        # `cumulative_gcd_count[g]` stores the total number of pairs whose GCD is less than or equal to `g`.
        # This is a prefix sum array of `exact_gcd_count`.
        # Time complexity: O(M).
        cumulative_gcd_count = [0] * (max_val + 1)
        current_sum = 0
        for g in range(1, max_val + 1):
            current_sum += exact_gcd_count[g]
            cumulative_gcd_count[g] = current_sum
        
        # Step 5: For each query, find the corresponding GCD value.
        # The `gcdPairs` array is sorted. A query `q` asks for `gcdPairs[q]`.
        # This means we need to find the smallest GCD value `G` such that there are `q + 1` or more pairs
        # with GCD less than or equal to `G`.
        # In other words, `cumulative_gcd_count[G]` must be greater than `q`.
        # `bisect.bisect_right(a, x)` returns the index `i` such that all `e` in `a[0:i]` have `e <= x`,
        # and all `e` in `a[i:]` have `e > x`. For our `cumulative_gcd_count` array, this `i` is precisely `G`.
        # Time complexity: O(Q log M).
        results = []
        # `cumulative_gcd_count[0]` is 0. This ensures `bisect_right` works correctly even for `query_val = 0`.
        for query_val in queries:
            ans_gcd = bisect.bisect_right(cumulative_gcd_count, query_val)
            results.append(ans_gcd)
            
        return results