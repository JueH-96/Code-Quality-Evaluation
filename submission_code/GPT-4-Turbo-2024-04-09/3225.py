class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # This will store the frequency of each element in the current window
        count = defaultdict(int)
        
        max_length = 0
        left = 0  # Left pointer for the sliding window
        
        # Right pointer for the sliding window
        for right in range(len(nums)):
            count[nums[right]] += 1
            
            # If the frequency of the current element exceeds k, shrink the window from the left
            while count[nums[right]] > k:
                count[nums[left]] -= 1
                left += 1
            
            # Update the maximum length of a good subarray
            max_length = max(max_length, right - left + 1)
        
        return max_length