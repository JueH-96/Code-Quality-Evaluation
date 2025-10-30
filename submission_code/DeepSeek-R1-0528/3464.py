class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        s0 = nums[0]
        s1 = -10**18
        
        for i in range(1, n):
            new_s0 = max(s0, s1) + nums[i]
            new_s1 = s0 - nums[i]
            s0, s1 = new_s0, new_s1
        
        return max(s0, s1)