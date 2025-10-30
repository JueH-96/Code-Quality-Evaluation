import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)

        # Handle edge case: array with one element.
        # If the array has only one element `x`, the score without removing is `x * x`.
        # If we remove the element, the array becomes empty, and the score of an empty array is 0.
        # Thus, the maximum score is `x * x`.
        if n == 1:
            return nums[0] * nums[0]

        # --- Preprocessing: Compute prefix and suffix GCD/LCM arrays ---
        # These arrays store the GCD/LCM of sub-arrays starting from the beginning
        # or ending at the end of the original array.
        # prefix_gcd[i] = gcd(nums[0], ..., nums[i])
        # suffix_gcd[i] = gcd(nums[i], ..., nums[n-1])
        # Similar definitions for prefix_lcm and suffix_lcm.
        prefix_gcd = [0] * n
        suffix_gcd = [0] * n
        prefix_lcm = [0] * n
        suffix_lcm = [0] * n

        # Calculate prefix arrays in a single pass from left to right
        prefix_gcd[0] = nums[0]
        prefix_lcm[0] = nums[0]
        for i in range(1, n):
            prefix_gcd[i] = math.gcd(prefix_gcd[i-1], nums[i])
            prefix_lcm[i] = math.lcm(prefix_lcm[i-1], nums[i])

        # Calculate suffix arrays in a single pass from right to left
        suffix_gcd[n-1] = nums[n-1]
        suffix_lcm[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix_gcd[i] = math.gcd(nums[i], suffix_gcd[i+1])
            suffix_lcm[i] = math.lcm(nums[i], suffix_lcm[i+1])

        # Helper functions to get prefix/suffix values safely, handling boundary cases.
        # When removing element at index i, we consider the elements before index i (nums[0]...nums[i-1])
        # and the elements after index i (nums[i+1]...nums[n-1]).
        # The GCD of the remaining elements is gcd(gcd(nums[0]...nums[i-1]), gcd(nums[i+1]...nums[n-1])).
        # The LCM of the remaining elements is lcm(lcm(nums[0]...nums[i-1]), lcm(nums[i+1]...nums[n-1])).
        # The GCD of an empty set is the identity for gcd, which is 0. gcd(0, x) = x for x > 0. gcd(0,0) = 0.
        def get_prefix_gcd(k):
            if k < 0: return 0 # Corresponds to the gcd of the empty slice nums[0...-1]
            return prefix_gcd[k]
        # The GCD of an empty set is 0.
        def get_suffix_gcd(k):
            if k >= n: return 0 # Corresponds to the gcd of the empty slice nums[n...n-1]
            return suffix_gcd[k]

        # The LCM of an empty set is the identity for lcm, which is 1. lcm(1, x) = x.
        def get_prefix_lcm(k):
            if k < 0: return 1 # Corresponds to the lcm of the empty slice nums[0...-1]
            return prefix_lcm[k]
        # The LCM of an empty set is 1.
        def get_suffix_lcm(k):
            if k >= n: return 1 # Corresponds to the lcm of the empty slice nums[n...n-1]
            return suffix_lcm[k]

        # --- Calculate maximum factor score ---
        # Initialize max_factor_score to 0, as scores are non-negative (product of numbers >= 1 or 0).
        max_factor_score = 0

        # Case 1: Do not remove any element.
        # The GCD of the entire array is suffix_gcd[0] (or prefix_gcd[n-1]).
        # The LCM of the entire array is suffix_lcm[0] (or prefix_lcm[n-1]).
        current_gcd = suffix_gcd[0]
        current_lcm = suffix_lcm[0]
        max_factor_score = max(max_factor_score, current_gcd * current_lcm)

        # Case 2: Remove one element at index i (0 <= i < n).
        # Since n > 1, removing one element leaves an array with n-1 >= 1 elements.
        for i in range(n):
            # Calculate the GCD of the elements remaining after removing nums[i].
            # This is the gcd of the prefix before index i and the suffix after index i.
            # gcd(remaining) = gcd(gcd(nums[0]...nums[i-1]), gcd(nums[i+1]...nums[n-1]))
            # We use helper functions to correctly get the prefix/suffix GCDs, handling the boundary
            # cases where one of the sub-arrays (before i or after i) is empty (when i=0 or i=n-1).
            temp_gcd = math.gcd(get_prefix_gcd(i-1), get_suffix_gcd(i+1))

            # Calculate the LCM of the elements remaining after removing nums[i].
            # This is the lcm of the prefix before index i and the suffix after index i.
            # lcm(remaining) = lcm(lcm(nums[0]...nums[i-1]), lcm(nums[i+1]...nums[n-1]))
            # We use helper functions to correctly get the prefix/suffix LCMs, handling the boundary
            # cases where one of the sub-arrays is empty.
            temp_lcm = math.lcm(get_prefix_lcm(i-1), get_suffix_lcm(i+1))

            # The product of the temporary GCD and LCM is the factor score
            # for the array with element at index i removed.
            # Update max_factor_score if the current score is greater.
            # Since n > 1, the remaining array is never empty, so temp_gcd >= 1 and temp_lcm >= 1.
            max_factor_score = max(max_factor_score, temp_gcd * temp_lcm)

        # Return the maximum factor score found across all cases (no removal or one removal).
        return max_factor_score