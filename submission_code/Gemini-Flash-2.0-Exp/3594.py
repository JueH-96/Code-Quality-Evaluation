class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        special_sum = sum(nums[:n-2])
        if special_sum == nums[n-2]:
            return nums[n-1]
        else:
            return nums[n-2]