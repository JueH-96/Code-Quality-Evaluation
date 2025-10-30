from collections import Counter

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        max_length = 0
        left = 0
        freq = Counter()
        
        for right in range(len(nums)):
            freq[nums[right]] += 1
            
            while any(count > k for count in freq.values()):
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length