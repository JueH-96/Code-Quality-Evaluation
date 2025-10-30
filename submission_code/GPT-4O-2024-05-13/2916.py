class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        
        # If the length of the array is 1, we cannot split it further
        if n == 1:
            return False
        
        # If the sum of the entire array is less than m, we cannot split it into valid subarrays
        if sum(nums) < m:
            return False
        
        # Check if we can split the array into n parts with the given condition
        for i in range(n - 1):
            if nums[i] + nums[i + 1] >= m:
                return True
        
        return False