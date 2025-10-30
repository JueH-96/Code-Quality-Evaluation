class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = 0
        curr_sum = 0
        left = 0
        for right in range(len(nums)):
            curr_sum += nums[right]
            while nums[right] - nums[left] > k:
                curr_sum -= nums[left]
                left += 1
            if nums[right] - nums[left] == k:
                max_sum = max(max_sum, curr_sum)
        return max_sum