from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        longest_seq_prefix_len = 1
        n = len(nums)
        for i in range(1, n):
            if nums[i] == nums[i - 1] + 1:
                longest_seq_prefix_len += 1
            else:
                break
        S = sum(nums[:longest_seq_prefix_len])
        nums_set = set(nums)
        x = S
        while x in nums_set:
            x += 1
        return x