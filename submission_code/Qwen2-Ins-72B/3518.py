from heapq import heappush, heappop
from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        a_pos = sorted((x for x in a if x > 0))
        a_neg = sorted((x for x in a if x < 0), reverse=True)
        
        # Calculate prefix sums for positive and negative elements
        a_pos_prefix = [0]
        for x in a_pos:
            a_pos_prefix.append(a_pos_prefix[-1] + x)
        
        a_neg_prefix = [0]
        for x in a_neg:
            a_neg_prefix.append(a_neg_prefix[-1] + x)
        
        # Calculate suffix sums for positive and negative elements
        a_pos_suffix = [0]
        for x in reversed(a_pos):
            a_pos_suffix.append(a_pos_suffix[-1] + x)
        a_pos_suffix.reverse()
        
        a_neg_suffix = [0]
        for x in reversed(a_neg):
            a_neg_suffix.append(a_neg_suffix[-1] + x)
        a_neg_suffix.reverse()
        
        # Calculate the maximum score
        max_score = float('-inf')
        pos_heap = []
        neg_heap = []
        
        for i in range(n):
            heappush(pos_heap, b[i])
            heappush(neg_heap, -b[i])
            
            if len(pos_heap) > len(a_pos):
                heappop(pos_heap)
            if len(neg_heap) > len(a_neg):
                heappop(neg_heap)
            
            if len(pos_heap) == len(a_pos) and len(neg_heap) == len(a_neg):
                pos_sum = sum(pos_heap)
                neg_sum = -sum(neg_heap)
                
                score = a_pos_prefix[len(pos_heap)] * pos_sum + a_neg_prefix[len(neg_heap)] * neg_sum
                score += a_pos_suffix[len(pos_heap)] * neg_sum + a_neg_suffix[len(neg_heap)] * pos_sum
                
                max_score = max(max_score, score)
        
        return max_score