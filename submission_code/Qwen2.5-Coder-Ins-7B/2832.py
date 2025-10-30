class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        left = 0
        max_freq = 0
        freq = defaultdict(int)
        
        for right in range(len(nums)):
            freq[nums[right]] += 1
            max_freq = max(max_freq, freq[nums[right]])
            
            if (right - left + 1) - max_freq > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
        
        return len(nums) - left