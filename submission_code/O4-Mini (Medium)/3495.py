from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # max-heap (as min-heap of negatives) to store the k smallest distances
        small = []
        # min-heap to store all other distances
        large = []
        results = []
        
        for x, y in queries:
            d = abs(x) + abs(y)
            # if we have fewer than k in 'small', just push
            if len(small) < k:
                heapq.heappush(small, -d)
            else:
                # if the new distance is smaller than the current k-th smallest
                # replace the k-th in small with this one, push the popped into large
                if d < -small[0]:
                    max_small = -heapq.heappushpop(small, -d)
                    heapq.heappush(large, max_small)
                else:
                    # otherwise it belongs to the larger half
                    heapq.heappush(large, d)
            
            # if we haven't reached k obstacles yet, answer is -1
            if len(small) < k:
                results.append(-1)
            else:
                # the k-th smallest is the max in 'small'
                results.append(-small[0])
        
        return results