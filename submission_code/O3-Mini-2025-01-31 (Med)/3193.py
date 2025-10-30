from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        n = len(nums)
        # Loop over all pairs (i, j) allowing same element and different indices
        for i in range(n):
            for j in range(i, n):
                x = nums[i]
                y = nums[j]
                # Check condition: |x-y| <= min(x,y)
                if abs(x - y) <= min(x, y):
                    current_xor = x ^ y
                    if current_xor > max_xor:
                        max_xor = current_xor
        return max_xor