import heapq
from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        # Initialize a min-heap with baskets sorted by their indices
        heap = []
        for idx, cap in enumerate(baskets):
            heapq.heappush(heap, (idx, cap))
        
        unplaced = 0
        
        for fruit in fruits:
            temp = []
            found = False
            # Find the first available basket with capacity >= fruit
            while heap:
                idx, cap = heapq.heappop(heap)
                if cap >= fruit:
                    found = True
                    break
                else:
                    temp.append((idx, cap))
            
            if not found:
                unplaced += 1
            else:
                # Push back the baskets that were popped but not used
                for item in temp:
                    heapq.heappush(heap, item)
        
        return unplaced