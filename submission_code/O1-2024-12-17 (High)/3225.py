class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        freq = defaultdict(int)
        left = 0
        max_len = 0
        
        for right in range(len(nums)):
            freq[nums[right]] += 1
            
            # Shrink the window while the current element's frequency exceeds k
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
            
            max_len = max(max_len, right - left + 1)
        
        return max_len