class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # If there's only one element, it's trivially special.
        if len(nums) == 1:
            return True
        
        # Iterate through adjacent pairs and check parity
        for i in range(1, len(nums)):
            if (nums[i] % 2) == (nums[i - 1] % 2):
                return False
        
        return True