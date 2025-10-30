class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        if len(nums) < k:
            return 0
        
        # Initialize the first window
        count = {}
        current_sum = 0
        
        # Build the first window of size k
        for i in range(k):
            current_sum += nums[i]
            count[nums[i]] = count.get(nums[i], 0) + 1
        
        # Check if first window is almost unique
        max_sum = 0
        if len(count) >= m:
            max_sum = current_sum
        
        # Slide the window
        for i in range(k, len(nums)):
            # Add new element
            current_sum += nums[i]
            count[nums[i]] = count.get(nums[i], 0) + 1
            
            # Remove element that's sliding out
            out_element = nums[i - k]
            current_sum -= out_element
            count[out_element] -= 1
            if count[out_element] == 0:
                del count[out_element]
            
            # Check if current window is almost unique
            if len(count) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum