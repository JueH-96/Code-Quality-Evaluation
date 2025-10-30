class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        diff = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            diff[i] += nums[i]
            diff[i+1] -= nums[i]
        
        for query in queries:
            l, r = query
            for i in range(l, r + 1):
                diff[i] -= 1
                diff[i+1] += 1
        
        for i in range(len(nums)):
            if diff[i] != 0:
                return False
        
        return True