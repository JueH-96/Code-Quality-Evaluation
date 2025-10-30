from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert numbers to strings for easy digit access
        str_nums = list(map(str, nums))
        n = len(str_nums)
        num_length = len(str_nums[0])

        # Initialize total difference
        total_diff = 0

        # Compare each pair of numbers
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate digit difference for the current pair
                diff = sum(a != b for a, b in zip(str_nums[i], str_nums[j]))
                total_diff += diff

        return total_diff