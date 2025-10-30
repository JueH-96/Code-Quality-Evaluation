class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = 0
        current_sum = 0
        start = 0

        for end in range(len(nums)):
            current_sum += nums[end]

            while abs(nums[end] - nums[start]) > k:
                current_sum -= nums[start]
                start += 1

            if abs(nums[end] - nums[start]) == k:
                max_sum = max(max_sum, current_sum)

        return max_sum