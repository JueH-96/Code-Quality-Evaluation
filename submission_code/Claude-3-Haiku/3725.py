from typing import List
from collections import deque

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Initialize the deques for minimum and maximum elements
        min_deque, max_deque = deque(), deque()
        
        # Initialize the total sum
        total_sum = 0
        
        for i in range(n):
            # Remove elements from the minimum deque that are out of the current window
            while min_deque and min_deque[0] < i - k + 1:
                min_deque.popleft()
            
            # Remove elements from the maximum deque that are out of the current window
            while max_deque and max_deque[0] < i - k + 1:
                max_deque.popleft()
            
            # Add the current element to the minimum and maximum deques
            while min_deque and nums[min_deque[-1]] >= nums[i]:
                min_deque.pop()
            min_deque.append(i)
            
            while max_deque and nums[max_deque[-1]] <= nums[i]:
                max_deque.pop()
            max_deque.append(i)
            
            # Add the minimum and maximum elements of the current window to the total sum
            if i >= k - 1:
                total_sum += nums[min_deque[0]] + nums[max_deque[0]]
        
        return total_sum