class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = {}
        left = 0
        max_len = 0
        
        for right in range(len(nums)):
            # Add current element to frequency map
            freq[nums[right]] = freq.get(nums[right], 0) + 1
            
            # While frequency of any element exceeds k
            # shrink window from left
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                left += 1
            
            # Update max length
            max_len = max(max_len, right - left + 1)
            
        return max_len