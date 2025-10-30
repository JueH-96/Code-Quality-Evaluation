class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        if len(nums) < k:
            return 0
        
        max_sum = 0
        current_sum = 0
        freq = {}
        
        # Initialize the first window
        for i in range(k):
            current_sum += nums[i]
            freq[nums[i]] = freq.get(nums[i], 0) + 1
        
        # Check if first window is almost unique
        if len(freq) >= m:
            max_sum = current_sum
        
        # Slide the window
        for i in range(k, len(nums)):
            # Add new element
            current_sum += nums[i]
            freq[nums[i]] = freq.get(nums[i], 0) + 1
            
            # Remove old element
            old_element = nums[i - k]
            current_sum -= old_element
            freq[old_element] -= 1
            if freq[old_element] == 0:
                del freq[old_element]
            
            # Check if current window is almost unique
            if len(freq) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum