from typing import List
import heapq

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = nums.length
        dp = [-float('inf')] * n
        dp[0] = 0
        max_heap = [(-nums[0], 0)]
        
        for i in range(1, n):
            while max_heap and max_heap[0][1] < i:
                heapq.heappop(max_heap)
            dp[i] = dp[max_heap[0][1]] + (i - max_heap[0][1]) * nums[max_heap[0][1]]
            heapq.heappush(max_heap, (-nums[i], i))
        
        return dp[-1]