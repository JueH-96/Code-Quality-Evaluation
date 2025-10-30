class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k:
            return 0
        
        max_sum = 0
        
        # Initialize the first window
        window_sum = sum(nums[:k])
        freq = {}
        for i in range(k):
            freq[nums[i]] = freq.get(nums[i], 0) + 1
        
        # Check if first window is almost unique
        if len(freq) >= m:
            max_sum = window_sum
        
        # Slide the window
        for i in range(k, n):
            # Add new element
            window_sum += nums[i]
            freq[nums[i]] = freq.get(nums[i], 0) + 1
            
            # Remove old element
            old_element = nums[i - k]
            window_sum -= old_element
            freq[old_element] -= 1
            if freq[old_element] == 0:
                del freq[old_element]
            
            # Check if current window is almost unique
            if len(freq) >= m:
                max_sum = max(max_sum, window_sum)
        
        return max_sum