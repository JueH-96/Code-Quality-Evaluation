import heapq
from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        queries.sort()
        max_heap = []
        j = 0
        cover = [0] * (n + 1)
        current_cover = 0
        total_selected = 0
        
        for i in range(n):
            current_cover += cover[i]
            
            while current_cover < nums[i]:
                # Add all queries with l <= i
                while j < len(queries) and queries[j][0] <= i:
                    # Push (-r, l, r)
                    heapq.heappush(max_heap, (-queries[j][1], queries[j][0], queries[j][1]))
                    j += 1
                
                # Remove invalid queries from the heap (r < i)
                while max_heap and max_heap[0][2] < i:
                    heapq.heappop(max_heap)
                
                if not max_heap:
                    return -1
                
                # Select the query with largest r
                selected = heapq.heappop(max_heap)
                selected_l = selected[1]
                selected_r = selected[2]
                
                # Update the cover array
                cover[selected_l] += 1
                cover[selected_r + 1] -= 1
                total_selected += 1
                current_cover += 1
                
        return len(queries) - total_selected