from typing import List
import heapq

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # Sort the array a to handle negative and positive values correctly
        a.sort()
        n = len(b)
        
        # Initialize min heaps for the smallest elements and max heaps for the largest elements
        min_heap = []
        max_heap = []
        
        # Populate the heaps with the first 4 elements of b
        for i in range(4):
            heapq.heappush(min_heap, -b[i])
            heapq.heappush(max_heap, b[i])
        
        # Iterate through the rest of the elements in b
        for i in range(4, n):
            if b[i] > min_heap[0]:
                heapq.heappushpop(min_heap, -b[i])
            if b[i] < max_heap[0]:
                heapq.heappushpop(max_heap, b[i])
        
        # Calculate the score using the top 4 elements from the heaps
        score = 0
        for i in range(4):
            if a[i] >= 0:
                score += a[i] * heapq.heappop(max_heap)
            else:
                score += a[i] * -heapq.heappop(min_heap)
        
        return score