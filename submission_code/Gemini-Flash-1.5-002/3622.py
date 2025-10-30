class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        left = 0
        total_ops = 0
        max_freq = 0
        for right in range(n):
            total_ops += (nums[right] - nums[left]) * (right - left)
            while total_ops > numOperations * (right-left+1):
                total_ops -= nums[right] - nums[left]
                left += 1

            max_freq = max(max_freq, right - left + 1)

        return max_freq