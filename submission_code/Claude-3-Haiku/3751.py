class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        left, right = 0, 0
        total = 0
        max_freq = 0

        while right < n:
            total += nums[right]

            while total + k < nums[right] * (right - left + 1):
                total -= nums[left]
                left += 1

            max_freq = max(max_freq, right - left + 1)
            right += 1

        return max_freq