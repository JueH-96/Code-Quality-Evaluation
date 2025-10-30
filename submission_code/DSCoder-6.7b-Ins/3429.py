class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        nums.sort()
        n = len(nums)
        if n < 3:
            return False
        if nums[0] >= nums[1] or nums[-1] <= nums[-2]:
            return False
        for i in range(1, n-1):
            if nums[i] == nums[i-1] or nums[i] == nums[i+1]:
                return False
            if nums[i] % 2 == nums[i-1] % 2 or nums[i] % 2 == nums[i+1] % 2:
                return False
        return True