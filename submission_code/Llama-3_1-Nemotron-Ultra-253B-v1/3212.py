from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        last = {}
        for i, x in enumerate(nums):
            last[x] = i
        block_count = 0
        current_end = 0
        for i in range(len(nums)):
            current_end = max(current_end, last[nums[i]])
            if i == current_end:
                block_count += 1
                current_end = 0  # Reset for the next block
        return pow(2, block_count - 1, MOD) if block_count else 0