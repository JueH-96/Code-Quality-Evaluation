class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]

        rightmost = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= rightmost:
                rightmost += nums[i]
            else:
                rightmost = nums[i]

        return rightmost