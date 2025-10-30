class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum_good_subarray = 0
        n = len(nums)
        for i in range(n):
            current_subarray_sum = 0
            for j in range(i, n):
                current_subarray_sum += nums[j]
                if j > i and abs(nums[i] - nums[j]) == k:
                    max_sum_good_subarray = max(max_sum_good_subarray, current_subarray_sum)

        return max_sum_good_subarray