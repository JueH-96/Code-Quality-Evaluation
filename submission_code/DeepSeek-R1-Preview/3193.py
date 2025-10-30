import bisect
from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        nums.sort()
        max_xor = 0
        n = len(nums)
        for i in range(n):
            target = 2 * nums[i]
            # Find the insertion point for target in the array starting from index i
            j = bisect.bisect_right(nums, target, i) - 1
            if j >= i:
                for k in range(i, j + 1):
                    current_xor = nums[i] ^ nums[k]
                    if current_xor > max_xor:
                        max_xor = current_xor
        return max_xor