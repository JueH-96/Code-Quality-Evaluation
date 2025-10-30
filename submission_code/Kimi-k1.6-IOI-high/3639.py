class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n + 1:
                diff[r + 1] -= 1
        
        coverage = 0
        for i in range(n):
            coverage += diff[i]
            if nums[i] > coverage:
                return False
        
        return True