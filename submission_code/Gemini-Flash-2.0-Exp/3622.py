class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        left = 0
        curr_operations = 0
        max_freq = 0
        for right in range(n):
            curr_operations += (nums[right] - nums[right - 1]) * (right - left) if right > 0 else 0
            while curr_operations > numOperations:
                curr_operations -= nums[right] - nums[left]
                left += 1
            max_freq = max(max_freq, right - left + 1)
        return max_freq