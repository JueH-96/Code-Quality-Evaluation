class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # If there's only one element, it's special by definition
        if len(nums) == 1:
            return True
        
        # Check each pair of adjacent elements
        return all(a % 2 != b % 2 for a, b in zip(nums, nums[1:]))