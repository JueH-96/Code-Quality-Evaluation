from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        # Initialize counts list with zeros
        counts = [0] * 51
        # Count the occurrences of each number
        for num in nums:
            counts[num] += 1
        # Collect numbers that appear exactly twice
        twice = [i for i in range(1, 51) if counts[i] == 2]
        # Compute the XOR of these numbers
        xor_result = 0
        for num in twice:
            xor_result ^= num
        return xor_result