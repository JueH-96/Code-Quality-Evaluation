class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * n
        for query in queries:
            l, r = query
            diff[l] += 1
            if r + 1 < n:
                diff[r + 1] -= 1
        
        cov_sum = 0
        for i in range(n):
            cov_sum += diff[i]
            if cov_sum < nums[i]:
                return False
        
        return True