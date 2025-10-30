from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # max-heap (via negatives) to store the k smallest distances seen so far
        small = []
        results = []
        
        for x, y in queries:
            d = abs(x) + abs(y)
            
            # If we haven't seen k obstacles yet, just add this one
            if len(small) < k:
                heapq.heappush(small, -d)
            else:
                # If we already have k, only insert if this one is closer
                if d < -small[0]:
                    # Replace the current largest in the k-smallest heap
                    heapq.heapreplace(small, -d)
            
            # If fewer than k obstacles, answer is -1
            if len(small) < k:
                results.append(-1)
            else:
                # The k-th nearest is the max in our max-heap
                results.append(-small[0])
        
        return results