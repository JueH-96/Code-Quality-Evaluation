class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(nums):
            max_so_far = nums[0]
            current_max = nums[0]
            for num in nums[1:]:
                current_max = max(num, current_max + num)
                max_so_far = max(max_so_far, current_max)
            return max_so_far

        def kadane_without_x(nums, x):
            max_so_far = None
            current_max = None
            for num in nums:
                if num == x:
                    current_max = None  # reset current_max
                else:
                    if current_max is None:
                        current_max = num
                    else:
                        current_max += num
                    if max_so_far is None or current_max > max_so_far:
                        max_so_far = current_max
            if max_so_far is None:
                return float('-inf')
            return max_so_far

        max_subarray_sum = kadane(nums)
        unique_nums = set(nums)
        max_sum = max_subarray_sum
        for x in unique_nums:
            max_sum_x = kadane_without_x(nums, x)
            max_sum = max(max_sum, max_sum_x)
        return max_sum