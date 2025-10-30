from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums: list[int], k: int) -> int:
        count = defaultdict(int)
        left = 0
        max_length = 0
        
        for right in range(len(nums)):
            count[nums[right]] += 1
            
            while count[nums[right]] > k:
                count[nums[left]] -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length