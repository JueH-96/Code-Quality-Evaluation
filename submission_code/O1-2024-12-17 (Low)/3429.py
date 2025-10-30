class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        for i in range(len(nums) - 1):
            # Check if the current and next elements have the same parity
            if (nums[i] % 2) == (nums[i+1] % 2):
                return False
        return True