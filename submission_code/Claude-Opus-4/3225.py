class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = {}
        left = 0
        max_length = 0
        
        for right in range(len(nums)):
            # Add current element to frequency map
            freq[nums[right]] = freq.get(nums[right], 0) + 1
            
            # If frequency exceeds k, shrink window from left
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
            
            # Update max length
            max_length = max(max_length, right - left + 1)
        
        return max_length