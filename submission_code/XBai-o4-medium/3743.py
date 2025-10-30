from typing import List
from collections import deque

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        gaps = []
        prev_end = 0
        for i in range(len(startTime)):
            current_start = startTime[i]
            gaps.append(current_start - prev_end)
            prev_end = endTime[i]
        gaps.append(eventTime - prev_end)
        
        m = len(gaps)
        prefix_sum = [0] * (m + 1)
        for i in range(m):
            prefix_sum[i + 1] = prefix_sum[i] + gaps[i]
        
        dq = deque()
        max_free = 0
        
        for i in range(m):
            # Remove elements out of the current window
            while dq and dq[0] < (i - k):
                dq.popleft()
            
            current_a = i
            # Maintain deque in increasing order of prefix_sum
            while dq and prefix_sum[current_a] <= prefix_sum[dq[-1]]:
                dq.pop()
            dq.append(current_a)
            
            # Calculate current maximum free time
            current_min = prefix_sum[dq[0]]
            current_sum = prefix_sum[i + 1] - current_min
            if current_sum > max_free:
                max_free = current_sum
        
        return max_free