from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        xor_result = 0
        for num in nums:
            xor_result ^= num
        return bin(xor_result ^ k).count('1')