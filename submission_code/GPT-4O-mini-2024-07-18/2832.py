class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        left = 0
        max_length = 0
        count = defaultdict(int)
        
        for right in range(len(nums)):
            count[nums[right]] += 1
            
            while (right - left + 1) - max(count.values()) > k:
                count[nums[left]] -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length