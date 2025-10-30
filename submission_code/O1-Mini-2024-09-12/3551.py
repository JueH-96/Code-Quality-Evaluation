from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        mask = [[] for _ in range(n+1)]
        for k in range(1, n+1):
            m = k-1
            for d in range(k):
                if (d & ~m) == 0:
                    mask[k].append(d)
        
        max_score = [[0]*n for _ in range(n)]
        for L in range(n):
            current_max = -1
            for R in range(L, n):
                k = R - L + 1
                s = 0
                for d in mask[k]:
                    s ^= nums[L + d]
                if s > current_max:
                    current_max = s
                max_score[L][R] = current_max
        
        res = []
        for q in queries:
            L, R = q
            res.append(max_score[L][R])
        return res