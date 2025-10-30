class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # If array has only one element, it's special by definition
        if len(nums) <= 1:
            return True
        
        # Check each adjacent pair
        for i in range(len(nums) - 1):
            # If adjacent elements have same parity, array is not special
            if nums[i] % 2 == nums[i + 1] % 2:
                return False
        
        return True