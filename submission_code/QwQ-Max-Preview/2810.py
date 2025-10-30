from typing import List
from collections import deque

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_total = float('inf')
        
        for m in range(1, n + 1):
            sum_min = 0
            arr = nums + nums  # Double the array to handle circularity
            dq = deque()
            
            for i in range(len(arr)):
                # Remove elements from the deque that are greater than the current element
                while dq and arr[i] <= arr[dq[-1]]:
                    dq.pop()
                dq.append(i)
                
                # Remove elements from the front that are out of the current window
                while dq[0] < i - m + 1:
                    dq.popleft()
                
                # Check if the current window is valid and within the original array
                if i >= m - 1:
                    start = i - m + 1
                    if start < n:
                        sum_min += arr[dq[0]]
                        # Early exit if all required windows are processed
                        if start == n - 1:
                            break
            
            # Calculate total cost for the current window size m
            total_cost = sum_min + x * (m - 1)
            if total_cost < min_total:
                min_total = total_cost
        
        return min_total