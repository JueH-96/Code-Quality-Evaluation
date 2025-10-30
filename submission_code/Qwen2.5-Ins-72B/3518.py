from typing import List
import heapq

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        pos = []
        neg = []
        
        for num in b:
            if num >= 0:
                heapq.heappush(pos, num)
            else:
                heapq.heappush(neg, -num)
        
        score = 0
        
        for i in range(4):
            if a[i] >= 0:
                if pos:
                    score += a[i] * heapq.heappop(pos)
                else:
                    score += a[i] * (-heapq.heappop(neg))
            else:
                if neg:
                    score += a[i] * (-heapq.heappop(neg))
                else:
                    score += a[i] * heapq.heappop(pos)
        
        return score