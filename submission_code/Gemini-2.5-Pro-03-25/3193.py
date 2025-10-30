import math
from typing import List

class Solution:
    """
    Solves the maximum strong pair XOR problem.
    """
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        """
        Finds the maximum XOR value among all strong pairs in the array nums.

        A pair (x, y) is strong if |x - y| <= min(x, y).

        Args:
            nums: A 0-indexed integer array. Constraints: 1 <= len(nums) <= 50, 1 <= nums[i] <= 100.

        Returns:
            The maximum XOR value out of all possible strong pairs.
        """
        n = len(nums)
        # Initialize max_xor to 0. 
        # Since we can pick the same integer twice (e.g., (x, x)), 
        # and |x - x| = 0 <= min(x, x) = x (because nums[i] >= 1),
        # pairs of the form (x, x) are always strong pairs.
        # Their XOR value is x ^ x = 0.
        # Therefore, the maximum XOR will always be at least 0.
        max_xor = 0

        # Iterate through all possible pairs of indices (i, j) from the nums array.
        # This covers all combinations, including picking the same element twice (when i == j)
        # and considering both (x, y) and (y, x).
        for i in range(n):
            for j in range(n):
                x = nums[i]
                y = nums[j]

                # Check if the pair (x, y) satisfies the strong pair condition.
                # The condition is |x - y| <= min(x, y).
                # abs() calculates the absolute difference.
                # min() finds the smaller of the two numbers.
                if abs(x - y) <= min(x, y):
                    # If the pair is strong, calculate its bitwise XOR using the ^ operator.
                    current_xor = x ^ y
                    # Update the maximum XOR value found so far if the current XOR is greater.
                    max_xor = max(max_xor, current_xor)

        # After checking all pairs, max_xor holds the maximum XOR value found among all strong pairs.
        # Return the overall maximum XOR value.
        return max_xor