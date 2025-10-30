class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        left = 0
        max_length = 0
        frequency = {}

        for right in range(len(nums)):
            frequency[nums[right]] = frequency.get(nums[right], 0) + 1

            while frequency[nums[right]] > k:
                frequency[nums[left]] -= 1
                left += 1

            max_length = max(max_length, right - left + 1)

        return max_length