class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)

        for i in range(n - 1, 1, -1):
            # Consider the triplet nums[i-2], nums[i-1], nums[i]
            # where nums[i] is the potential longest side
            if nums[i] < nums[i - 1] + nums[i - 2]:
                return nums[i - 2] + nums[i - 1] + nums[i]

        return -1