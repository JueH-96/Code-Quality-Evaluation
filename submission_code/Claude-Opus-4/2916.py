class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        
        # Base cases
        if n <= 2:
            return True
        
        # For n >= 3, check if there exists at least one pair of adjacent elements
        # with sum >= m
        for i in range(n - 1):
            if nums[i] + nums[i + 1] >= m:
                return True
        
        return False