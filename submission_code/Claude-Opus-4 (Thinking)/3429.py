class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # If array has only one element, it's special
        if len(nums) == 1:
            return True
        
        # Check all adjacent pairs
        for i in range(len(nums) - 1):
            # Check if both have same parity
            if nums[i] % 2 == nums[i+1] % 2:
                return False
        
        return True