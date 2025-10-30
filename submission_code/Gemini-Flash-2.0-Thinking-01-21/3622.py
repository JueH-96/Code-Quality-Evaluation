class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        left = 0
        current_sum = 0
        max_freq = 1
        for right in range(n):
            current_sum += nums[right]
            while (right - left + 1) * nums[right] - current_sum > numOperations:
                current_sum -= nums[left]
                left += 1
            max_freq = max(max_freq, right - left + 1)
        return max_freq