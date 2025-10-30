class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = {}
        left = 0
        max_length = 0
        
        for right in range(len(nums)):
            if nums[right] in freq:
                freq[nums[right]] += 1
            else:
                freq[nums[right]] = 1
            
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length