from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_count = [0] * n
        seen_pre = set()
        for i in range(n):
            seen_pre.add(nums[i])
            prefix_count[i] = len(seen_pre)
        
        suffix_count = [0] * (n + 1)
        seen_suf = set()
        for i in range(n - 1, -1, -1):
            seen_suf.add(nums[i])
            suffix_count[i] = len(seen_suf)
        
        res = []
        for i in range(n):
            res.append(prefix_count[i] - suffix_count[i + 1])
        
        return res