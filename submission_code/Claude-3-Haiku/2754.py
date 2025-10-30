class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # Initialize the maximum and minimum product up to each index
        max_prod = [0] * n
        min_prod = [0] * n
        
        # Calculate the maximum and minimum product up to each index
        max_prod[0] = nums[0]
        min_prod[0] = nums[0]
        for i in range(1, n):
            max_prod[i] = max(nums[i], max_prod[i-1] * nums[i], min_prod[i-1] * nums[i])
            min_prod[i] = min(nums[i], max_prod[i-1] * nums[i], min_prod[i-1] * nums[i])
        
        # Find the maximum strength
        return max(max_prod)