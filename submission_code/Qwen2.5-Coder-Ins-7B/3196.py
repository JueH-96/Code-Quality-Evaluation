class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        max_freq = 0
        total_cost = 0
        
        for right in range(len(nums)):
            total_cost += (nums[right] - nums[right - 1]) * (right - left)
            while total_cost > k:
                total_cost -= nums[right] - nums[left]
                left += 1
            max_freq = max(max_freq, right - left + 1)
        
        return max_freq