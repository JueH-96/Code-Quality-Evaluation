class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        return min(nums[0] + nums[i] + nums[j] 
                   for i in range(1, n-1) 
                   for j in range(i+1, n))