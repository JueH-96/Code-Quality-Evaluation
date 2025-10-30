from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        current_sum = 0
        current_elements = {}
        
        for i in range(len(nums)):
            # Add the current element to the current sum and update the count in the dictionary
            current_sum += nums[i]
            current_elements[nums[i]] = current_elements.get(nums[i], 0) + 1
            
            # If the window size exceeds k, remove the element that is out of the window
            if i >= k:
                current_sum -= nums[i - k]
                current_elements[nums[i - k]] -= 1
                if current_elements[nums[i - k]] == 0:
                    del current_elements[nums[i - k]]
            
            # Check if the current window is almost unique
            if i >= k - 1 and len(current_elements) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum