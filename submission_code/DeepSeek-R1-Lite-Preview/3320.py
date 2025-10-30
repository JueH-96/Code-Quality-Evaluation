class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        target = nums[0] + nums[1]
        count = 0
        while len(nums) >= 2:
            if nums[0] + nums[1] == target:
                count += 1
                nums.pop(0)
                nums.pop(0)
            else:
                break
        return count