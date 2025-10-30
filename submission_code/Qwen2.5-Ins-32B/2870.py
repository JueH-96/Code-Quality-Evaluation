from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if nums[j] - nums[j - 1] == (-1) ** (j - i):
                    continue
                else:
                    if j - i > 1:
                        max_len = max(max_len, j - i)
                    break
            else:
                if j - i + 1 > 1:
                    max_len = max(max_len, j - i + 1)
        
        return max_len