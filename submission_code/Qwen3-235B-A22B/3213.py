import bisect
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        M = max(nums)
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            if nums[i - 1] == M:
                prefix[i] = prefix[i - 1] + 1
            else:
                prefix[i] = prefix[i - 1]
        
        res = 0
        for j in range(1, n + 1):
            current_sum = prefix[j]
            target = current_sum - k
            if target < 0:
                continue
            count = bisect.bisect_right(prefix, target, 0, j)
            res += count
        
        return res