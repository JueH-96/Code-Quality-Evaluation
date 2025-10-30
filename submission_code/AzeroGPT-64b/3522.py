from collections import deque
from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        left = 0
        max_deque = deque()
        
        for right in range(n):
            # Ensure deque is always decreasing
            while max_deque and nums[max_deque[-1]] <= nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # Remove elements outside the current window
            if left > max_deque[0]:
                max_deque.popleft()
                
            # Check if window size is k and if array is sorted
            if right - left + 1 == k:
                current_subarray = nums[left:right+1]
                if all(current_subarray[i] < current_subarray[i+1] for i in range(len(current_subarray)-1)):
                    results.append(nums[max_deque[0]])
                else:
                    results.append(-1)
                left += 1
                
        return results