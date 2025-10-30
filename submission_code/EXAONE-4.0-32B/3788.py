from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        ans = max(nums)
        for i in range(n):
            seen = set()
            cur_sum = 0
            for j in range(i, n):
                x = nums[j]
                if x > 0 and x not in seen:
                    seen.add(x)
                    cur_sum += x
                if cur_sum > 0:
                    if cur_sum > ans:
                        ans = cur_sum
        return ans