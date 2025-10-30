from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pre = [0] * n
        suf = [0] * n
        
        pre[0] = nums[0]
        for i in range(1, n):
            pre[i] = pre[i-1] | nums[i]
        
        suf[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suf[i] = suf[i+1] | nums[i]
        
        max_or = 0
        for i in range(n):
            if i == 0:
                others = suf[1] if n > 1 else 0
            elif i == n-1:
                others = pre[n-2] if n > 1 else 0
            else:
                others = pre[i-1] | suf[i+1]
            candidate = (nums[i] << k) | others
            max_or = max(max_or, candidate)
        
        return max_or