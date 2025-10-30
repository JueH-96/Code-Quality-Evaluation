from collections import deque
from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        deques = [deque() for _ in range(k)]
        max_sum = float('-inf')
        
        for a in range(n):
            current_prefix = prefix_sum[a + 1]
            r_b = (a + 1) % k
            
            # Remove elements from the end of the deque that are greater than (a+1 -k)
            while deques[r_b] and deques[r_b][-1] > (a + 1 - k):
                deques[r_b].pop()
            
            if deques[r_b]:
                current_sum = current_prefix - deques[r_b][0]
                if current_sum > max_sum:
                    max_sum = current_sum
            
            r = a % k
            # Maintain the deque in increasing order of prefix_sum
            while deques[r] and prefix_sum[deques[r][-1]] >= prefix_sum[a]:
                deques[r].pop()
            deques[r].append(a)
        
        return max_sum if max_sum != float('-inf') else 0