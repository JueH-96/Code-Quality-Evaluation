class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        freq = defaultdict(int)  # To keep track of frequencies in the current window
        left = 0
        max_freq = 0  # The maximum frequency of any element in the current window
        result = 0
        
        for right in range(len(nums)):
            # Include nums[right] in the sliding window
            freq[nums[right]] += 1
            
            # Update the maximum frequency for the current window
            max_freq = max(max_freq, freq[nums[right]])
            
            # If the cost (window_size - max_freq) to remove non-frequent elements exceeds k,
            # shrink the window from the left
            window_size = right - left + 1
            while window_size - max_freq > k:
                freq[nums[left]] -= 1
                left += 1
                window_size -= 1
            
            # The final subarray (all equal) in this window has length = max_freq
            result = max(result, max_freq)
        
        return result