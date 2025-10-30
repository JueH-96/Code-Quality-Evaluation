class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        for i in range(len(nums) - 1):
            # If consecutive elements have the same parity, return False
            if (nums[i] % 2) == (nums[i+1] % 2):
                return False
        return True