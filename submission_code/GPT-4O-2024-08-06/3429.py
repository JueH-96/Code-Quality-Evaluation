class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # If the array has only one element, it is considered special
        if len(nums) == 1:
            return True
        
        # Iterate through the array and check each pair of adjacent elements
        for i in range(len(nums) - 1):
            # Check if both elements have the same parity
            if (nums[i] % 2) == (nums[i + 1] % 2):
                return False
        
        # If no adjacent elements have the same parity, the array is special
        return True