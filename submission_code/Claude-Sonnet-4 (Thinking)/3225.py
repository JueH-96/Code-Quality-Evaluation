class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        left = 0
        max_len = 0
        freq = {}
        
        for right in range(len(nums)):
            # Add the current element to the window
            freq[nums[right]] = freq.get(nums[right], 0) + 1
            
            # Shrink the window if any frequency exceeds k
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
            
            # Update the maximum length
            max_len = max(max_len, right - left + 1)
        
        return max_len