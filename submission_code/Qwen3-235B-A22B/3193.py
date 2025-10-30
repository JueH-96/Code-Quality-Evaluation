from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        for x in nums:
            for y in nums:
                if abs(x - y) <= min(x, y):
                    current_xor = x ^ y
                    if current_xor > max_xor:
                        max_xor = current_xor
        return max_xor