from typing import List
from collections import deque

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        dq = deque()
        dq.append((nums[0], 0))  # Initial line for index 0
        
        max_score = 0
        
        for i in range(1, n):
            # Query phase: find the best line at x=i
            while len(dq) >= 2:
                a1, b1 = dq[0]
                a2, b2 = dq[1]
                val1 = a1 * i + b1
                val2 = a2 * i + b2
                if val2 > val1:
                    dq.popleft()
                else:
                    break
            
            a, b = dq[0]
            current = a * i + b
            max_score = current  # Update the max_score as we go
            
            # Insertion phase: add the new line (nums[i], current - i * nums[i])
            new_a = nums[i]
            new_b = current - i * new_a
            
            # Maintain the deque by removing lines that are no longer useful
            while len(dq) >= 2:
                a2, b2 = dq[-1]
                a1, b1 = dq[-2]
                lhs = (new_b - b2) * (a1 - a2)
                rhs = (b2 - b1) * (a2 - new_a)
                if lhs <= rhs:
                    dq.pop()
                else:
                    break
            dq.append((new_a, new_b))
        
        return max_score