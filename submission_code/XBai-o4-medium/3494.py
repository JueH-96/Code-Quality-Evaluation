from typing import List
import heapq

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Create max heaps for horizontal and vertical cuts
        h_heap = [-h for h in horizontalCut]
        heapq.heapify(h_heap)
        
        v_heap = [-v for v in verticalCut]
        heapq.heapify(v_heap)
        
        h_count = 0  # number of horizontal cuts made
        v_count = 0  # number of vertical cuts made
        total = 0
        
        while h_heap or v_heap:
            # Get current maximum horizontal and vertical cuts
            current_h = -h_heap[0] if h_heap else 0
            current_v = -v_heap[0] if v_heap else 0
            
            if current_h > current_v:
                # Perform horizontal cut
                max_h = -heapq.heappop(h_heap)
                total += max_h * (v_count + 1)
                h_count += 1
            else:
                # Perform vertical cut
                max_v = -heapq.heappop(v_heap)
                total += max_v * (h_count + 1)
                v_count += 1
        
        return total