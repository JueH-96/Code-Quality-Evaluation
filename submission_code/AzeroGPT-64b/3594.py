class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        nums = sorted(nums)
        cum_sum = list(accumulate(nums))
        largest_outlier = None
        for i in range(2, len(nums)):
            sum_val = cum_sum[i - 1] if i > 0 else 0
            if (sum_val * 2 == nums[i] + cum_sum[-1]) and nums[i] not in nums[:i]:
                largest_outlier = max(largest_outlier, nums[-1] if i + 1 < len(nums) else nums[i])
            if (sum_val == nums[-1] + cum_sum[-1] - cum_sum[i]) and nums[-1] not in nums[:i]:
                largest_outlier = max(largest_outlier, nums[i] if i + 1 < len(nums) else nums[-1])
        return largest_outlier