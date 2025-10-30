class Solution:
    def isGood(self, nums: List[int]) -> bool:
        nums.sort()
        for i in range(len(nums)):
            if i != nums[i] - 1:
                return False
        return True