class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        max_freq = 0
        current_min = nums[0]
        current_max = nums[0]
        current_k = 0
        
        for right in range(len(nums)):
            current_max = max(current_max, nums[right])
            current_k += (nums[right] - current_min) * (right - left + 1) - (current_max - nums[right]) * (right - left)
            
            while current_k > k:
                current_k -= (current_max - nums[left]) * (right - left) + (current_max - nums[left + 1]) * (right - left)
                current_min = nums[left + 1]
                left += 1
                
            max_freq = max(max_freq, right - left + 1)
            
        return max_freq