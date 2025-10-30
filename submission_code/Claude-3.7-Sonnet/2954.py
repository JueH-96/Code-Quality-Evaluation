class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        
        # Initialize the first window
        window_counter = {}
        window_sum = 0
        for i in range(k):
            window_counter[nums[i]] = window_counter.get(nums[i], 0) + 1
            window_sum += nums[i]
        
        # Check if the first window has at least m distinct elements
        if len(window_counter) >= m:
            max_sum = window_sum
        
        # Slide the window
        for i in range(k, len(nums)):
            # Add the new element to the window
            window_counter[nums[i]] = window_counter.get(nums[i], 0) + 1
            window_sum += nums[i]
            
            # Remove the leftmost element from the window
            window_counter[nums[i - k]] -= 1
            if window_counter[nums[i - k]] == 0:
                del window_counter[nums[i - k]]
            window_sum -= nums[i - k]
            
            # Check if the window has at least m distinct elements
            if len(window_counter) >= m:
                max_sum = max(max_sum, window_sum)
        
        return max_sum