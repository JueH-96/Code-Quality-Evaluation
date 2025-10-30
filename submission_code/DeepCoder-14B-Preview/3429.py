class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True
        for i in range(len(nums) - 1):
            current = nums[i]
            next_num = nums[i + 1]
            if (current % 2) == (next_num % 2):
                return False
        return True