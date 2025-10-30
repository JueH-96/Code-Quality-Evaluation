class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        
        # Arrays of length 1 or 2 can always be split
        if n <= 2:
            return True
        
        # For arrays of length > 2, need at least one adjacent pair with sum >= m
        for i in range(n-1):
            if nums[i] + nums[i+1] >= m:
                return True
        
        return False