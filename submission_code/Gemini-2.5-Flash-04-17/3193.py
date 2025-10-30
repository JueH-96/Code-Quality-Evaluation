from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        # Initialize the maximum XOR value found so far to 0.
        # The minimum possible XOR value is 0 (e.g., x ^ x = 0 for any x).
        maxXor = 0

        # Get the number of elements in the array.
        n = len(nums)

        # Iterate through all possible pairs of integers from the array nums.
        # We need to consider all combinations of two integers from the array,
        # including selecting the same integer twice. A simple way to do this
        # is using nested loops iterating through the indices.
        for i in range(n):
            x = nums[i] # Get the first number of the pair
            for j in range(n):
                y = nums[j] # Get the second number of the pair

                # Check if the pair (x, y) satisfies the strong pair condition:
                # |x - y| <= min(x, y).
                # The condition is symmetric for x and y.
                if abs(x - y) <= min(x, y):
                    # If it is a strong pair, calculate the bitwise XOR of the two numbers.
                    currentXor = x ^ y
                    # Update the maximum XOR value found so far if the current XOR is greater.
                    maxXor = max(maxXor, currentXor)

        # After checking all possible pairs, return the maximum XOR value found among strong pairs.
        return maxXor