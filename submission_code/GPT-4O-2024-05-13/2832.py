from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        max_len = 0
        left = 0
        count = defaultdict(int)
        max_count = 0
        
        for right in range(len(nums)):
            count[nums[right]] += 1
            max_count = max(max_count, count[nums[right]])
            
            while (right - left + 1) - max_count > k:
                count[nums[left]] -= 1
                left += 1
            
            max_len = max(max_len, right - left + 1)
        
        return max_len