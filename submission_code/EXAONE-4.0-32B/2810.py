from collections import deque
from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        B = nums + nums
        ans = float('inf')
        
        for M in range(0, n):
            dq = deque()
            for j in range(n - M, n + 1):
                while dq and B[dq[-1]] > B[j]:
                    dq.pop()
                dq.append(j)
            
            total_sum = B[dq[0]]
            
            for s in range(n - M + 1, 2 * n - 1 - M + 1):
                while dq and dq[0] < s:
                    dq.popleft()
                
                j = s + M
                while dq and B[dq[-1]] > B[j]:
                    dq.pop()
                dq.append(j)
                
                total_sum += B[dq[0]]
            
            candidate = M * x + total_sum
            if candidate < ans:
                ans = candidate
        
        return ans