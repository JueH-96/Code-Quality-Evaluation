import heapq
from typing import List

class Solution:
    def kthDistance(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []
        
        for x, y in queries:
            # Calculate the distance of the new obstacle from the origin
            distance = abs(x) + abs(y)
            
            # Add the new obstacle to the heap
            heapq.heappush(obstacles, distance)
            
            # If there are less than k obstacles, append -1 to the results
            if len(obstacles) < k:
                results.append(-1)
            else:
                # Remove the smallest distance from the heap k-1 times
                temp = []
                for _ in range(k - 1):
                    heapq.heappush(temp, heapq.heappop(obstacles))
                # The kth smallest distance is now at the top of the heap
                kth_distance = obstacles[0]
                # Push the removed distances back into the heap
                while temp:
                    heapq.heappush(obstacles, heapq.heappop(temp))
                # Append the kth distance to the results
                results.append(kth_distance)
        
        return results