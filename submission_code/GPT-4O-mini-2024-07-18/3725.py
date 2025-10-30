from typing import List
from collections import deque

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_sum = 0
        
        # Calculate the sum of max and min for all subarrays of size 1 to k
        for size in range(1, k + 1):
            max_deque = deque()
            min_deque = deque()
            
            for i in range(n):
                # Remove elements not within the window
                if max_deque and max_deque[0] < i - size + 1:
                    max_deque.popleft()
                if min_deque and min_deque[0] < i - size + 1:
                    min_deque.popleft()
                
                # Maintain the max deque
                while max_deque and nums[max_deque[-1]] <= nums[i]:
                    max_deque.pop()
                max_deque.append(i)
                
                # Maintain the min deque
                while min_deque and nums[min_deque[-1]] >= nums[i]:
                    min_deque.pop()
                min_deque.append(i)
                
                # If we have filled at least 'size' elements, we can calculate the sum
                if i >= size - 1:
                    total_sum += nums[max_deque[0]] + nums[min_deque[0]]
        
        return total_sum