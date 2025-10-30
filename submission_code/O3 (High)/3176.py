from typing import List
import math

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        INF = math.inf
        
        # leftMin[j] = smallest value to the left of j that is < nums[j]
        leftMin = [INF] * n
        for j in range(n):
            best = INF
            for i in range(j):
                if nums[i] < nums[j]:
                    best = min(best, nums[i])
            leftMin[j] = best
        
        # rightMin[j] = smallest value to the right of j that is < nums[j]
        rightMin = [INF] * n
        for j in range(n):
            best = INF
            for k in range(j + 1, n):
                if nums[k] < nums[j]:
                    best = min(best, nums[k])
            rightMin[j] = best
        
        # compute minimal possible sum
        ans = INF
        for j in range(n):
            if leftMin[j] != INF and rightMin[j] != INF:
                ans = min(ans, leftMin[j] + nums[j] + rightMin[j])
        
        return -1 if ans == INF else ans