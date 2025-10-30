from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        # Dictionary to count occurrences of each number
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1

        # Variable to store the XOR of numbers that appear twice
        result = 0
        # XOR all numbers that appear exactly twice
        for num, freq in count.items():
            if freq == 2:
                result ^= num

        return result