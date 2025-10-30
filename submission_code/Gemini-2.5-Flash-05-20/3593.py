import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)

        # Helper function to calculate LCM of two numbers.
        # This function assumes inputs a, b are positive integers.
        # Constraints (1 <= nums[i]) ensure positive inputs, so a and b will always be >= 1.
        # Thus, math.gcd(a, b) will always be >= 1.
        # Python's integers handle arbitrary size, so no overflow concerns for a*b.
        def calculate_lcm(a, b):
            return (a * b) // math.gcd(a, b)

        # Base case: If only one element,
        # - Not removing it gives score nums[0] * nums[0].
        # - Removing it makes array empty, score is 0.
        # The maximum score will be nums[0] * nums[0] as nums[0] >= 1.
        if n == 1:
            return nums[0] * nums[0]

        # Precompute prefix GCDs and LCMs
        # prefix_gcd[i] stores gcd(nums[0], ..., nums[i])
        # prefix_lcm[i] stores lcm(nums[0], ..., nums[i])
        prefix_gcd = [0] * n
        prefix_lcm = [0] * n

        prefix_gcd[0] = nums[0]
        prefix_lcm[0] = nums[0]
        for i in range(1, n):
            prefix_gcd[i] = math.gcd(prefix_gcd[i-1], nums[i])
            prefix_lcm[i] = calculate_lcm(prefix_lcm[i-1], nums[i])

        # Precompute suffix GCDs and LCMs
        # suffix_gcd[i] stores gcd(nums[i], ..., nums[n-1])
        # suffix_lcm[i] stores lcm(nums[i], ..., nums[n-1])
        suffix_gcd = [0] * n
        suffix_lcm = [0] * n

        suffix_gcd[n-1] = nums[n-1]
        suffix_lcm[n-1] = nums[n-1]
        for i in range(n - 2, -1, -1): # Iterate from n-2 down to 0
            suffix_gcd[i] = math.gcd(suffix_gcd[i+1], nums[i])
            suffix_lcm[i] = calculate_lcm(suffix_lcm[i+1], nums[i])

        # Initialize max_factor_score with the score of the original array (no removal)
        # This covers the "at most one element" requirement for zero removals.
        max_factor_score = prefix_gcd[n-1] * prefix_lcm[n-1]

        # Iterate through all possibilities of removing one element
        for i in range(n):
            current_gcd_after_removal = 0
            current_lcm_after_removal = 0

            if i == 0:
                # Removed nums[0]. Remaining elements are nums[1]...nums[n-1].
                # Their GCD/LCM is simply suffix_gcd[1]/suffix_lcm[1].
                current_gcd_after_removal = suffix_gcd[1]
                current_lcm_after_removal = suffix_lcm[1]
            elif i == n - 1:
                # Removed nums[n-1]. Remaining elements are nums[0]...nums[n-2].
                # Their GCD/LCM is simply prefix_gcd[n-2]/prefix_lcm[n-2].
                current_gcd_after_removal = prefix_gcd[n-2]
                current_lcm_after_removal = prefix_lcm[n-2]
            else:
                # Removed nums[i]. Remaining elements are nums[0]...nums[i-1]
                # combined with nums[i+1]...nums[n-1].
                # GCD is gcd(prefix_gcd[i-1], suffix_gcd[i+1]).
                # LCM is lcm(prefix_lcm[i-1], suffix_lcm[i+1]).
                current_gcd_after_removal = math.gcd(prefix_gcd[i-1], suffix_gcd[i+1])
                current_lcm_after_removal = calculate_lcm(prefix_lcm[i-1], suffix_lcm[i+1])
            
            # Update the maximum score found so far
            max_factor_score = max(max_factor_score, current_gcd_after_removal * current_lcm_after_removal)
        
        return max_factor_score