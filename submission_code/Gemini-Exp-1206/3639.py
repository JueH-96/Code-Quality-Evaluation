class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        for l, r in queries:
            diff[l] += 1
            diff[r + 1] -= 1
        
        curr = 0
        for i in range(n):
            curr += diff[i]
            nums[i] -= curr
        
        for num in nums:
            if num < 0:
                return False
        
        return all(num == 0 for num in nums)