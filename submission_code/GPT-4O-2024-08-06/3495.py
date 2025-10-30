class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        import heapq
        
        # Min-heap to store the distances
        min_heap = []
        results = []
        
        for x, y in queries:
            # Calculate the Manhattan distance from the origin
            distance = abs(x) + abs(y)
            # Add the distance to the heap
            heapq.heappush(min_heap, distance)
            
            # If we have at least k obstacles, get the k-th smallest distance
            if len(min_heap) >= k:
                # Get the k-th smallest distance by sorting the heap and accessing the k-1 index
                results.append(sorted(min_heap)[k-1])
            else:
                # If there are less than k obstacles, append -1
                results.append(-1)
        
        return results