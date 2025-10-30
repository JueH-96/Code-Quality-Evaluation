class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if k > n:
            return 0
        
        max_sum = 0
        current_sum = sum(nums[:k])
        distinct_count = len(set(nums[:k]))
        
        window = {}
        for i in range(k):
            window[nums[i]] = window.get(nums[i], 0) + 1
        
        if distinct_count >= m:
            max_sum = current_sum
        
        for i in range(k, n):
            # Remove the leftmost element from the window
            window[nums[i-k]] -= 1
            if window[nums[i-k]] == 0:
                del window[nums[i-k]]
            
            # Add the new element to the window
            window[nums[i]] = window.get(nums[i], 0) + 1
            
            # Update the current sum
            current_sum = current_sum - nums[i-k] + nums[i]
            
            # Check if the current window is almost unique
            if len(window) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum