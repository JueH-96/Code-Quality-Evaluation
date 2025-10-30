from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        """
        Finds the maximum XOR value among all possible strong pairs in the given array nums.

        A pair of integers (x, y) is called a strong pair if it satisfies the condition:
        |x - y| <= min(x, y)

        Args:
            nums: A 0-indexed integer array.

        Returns:
            The maximum XOR value out of all possible strong pairs in the array nums.
        """

        max_xor_val = 0  # Initialize with 0. The XOR of any number with itself is 0,
                         # which is the minimum possible non-negative XOR value.

        n = len(nums)

        # Iterate through all possible pairs (x, y) from nums.
        # We use nested loops to select two numbers, nums[i] and nums[j].
        # Since the problem allows picking the same integer twice (i.e., i can be equal to j),
        # and the order of x and y doesn't affect the strong pair condition or XOR result (x^y == y^x),
        # iterating through all combinations of indices (i, j) covers all necessary pairs.
        for i in range(n):
            x = nums[i]  # First number in the pair
            for j in range(n):
                y = nums[j]  # Second number in the pair

                # Check if (x, y) is a strong pair based on the given condition:
                # |x - y| <= min(x, y)
                # Since nums[i] are constrained to be between 1 and 100, min(x, y) will always be at least 1.
                if abs(x - y) <= min(x, y):
                    # If it's a strong pair, calculate their bitwise XOR
                    current_xor = x ^ y
                    
                    # Update the maximum XOR value found so far
                    if current_xor > max_xor_val:
                        max_xor_val = current_xor
        
        return max_xor_val