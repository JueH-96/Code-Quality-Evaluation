class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        max_freq = 1
        total = 0
        
        for right in range(1, len(nums)):
            total += (nums[right] - nums[right - 1]) * (right - left)
            
            while total > k:
                total -= nums[right] - nums[left]
                left += 1
            
            max_freq = max(max_freq, right - left + 1)
        
        return max_freq