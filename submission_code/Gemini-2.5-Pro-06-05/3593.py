import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        """
        Calculates the maximum factor score of nums after removing at most one element.
        The factor score is the product of the LCM and GCD of an array's elements.
        """

        # Helper function to compute the Least Common Multiple (LCM) of two numbers.
        def lcm(a, b):
            # All numbers in nums are >= 1, so a and b will be positive.
            if a == 0 or b == 0:
                return 0
            return (a * b) // math.gcd(a, b)

        n = len(nums)

        # If there's only one element, the score is num*num (for keeping it)
        # or 0 (for removing it). The max is num*num.
        if n == 1:
            return nums[0] * nums[0]

        # O(N) precomputation of prefix and suffix GCDs and LCMs.
        prefix_gcd = [0] * n
        prefix_lcm = [0] * n
        suffix_gcd = [0] * n
        suffix_lcm = [0] * n

        # Fill prefix arrays
        prefix_gcd[0] = nums[0]
        prefix_lcm[0] = nums[0]
        for i in range(1, n):
            prefix_gcd[i] = math.gcd(prefix_gcd[i - 1], nums[i])
            prefix_lcm[i] = lcm(prefix_lcm[i - 1], nums[i])

        # Fill suffix arrays
        suffix_gcd[n - 1] = nums[n - 1]
        suffix_lcm[n - 1] = nums[n - 1]
        for i in range(n - 2, -1, -1):
            suffix_gcd[i] = math.gcd(suffix_gcd[i + 1], nums[i])
            suffix_lcm[i] = lcm(suffix_lcm[i + 1], nums[i])

        # 1. Calculate the score for the original array (no removal).
        # The full array's GCD is prefix_gcd[n-1] and LCM is prefix_lcm[n-1].
        max_score = prefix_gcd[n - 1] * prefix_lcm[n - 1]

        # 2. Iterate through each element, calculate score if it's removed.
        for i in range(n):
            current_gcd = 0
            current_lcm = 1 # Identity for LCM is 1

            if i == 0:
                # Removed the first element, use the suffix of the rest.
                current_gcd = suffix_gcd[1]
                current_lcm = suffix_lcm[1]
            elif i == n - 1:
                # Removed the last element, use the prefix of the rest.
                current_gcd = prefix_gcd[n - 2]
                current_lcm = prefix_lcm[n - 2]
            else:
                # Removed a middle element, combine prefix and suffix parts.
                current_gcd = math.gcd(prefix_gcd[i - 1], suffix_gcd[i + 1])
                current_lcm = lcm(prefix_lcm[i - 1], suffix_lcm[i + 1])
            
            # Update the maximum score found so far.
            max_score = max(max_score, current_gcd * current_lcm)
            
        return max_score