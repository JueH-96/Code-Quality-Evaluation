class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        import heapq
        
        # max_heap will store up to the k smallest distances (as negative values)
        max_heap = []
        # min_heap will store the distances greater than those in max_heap
        min_heap = []
        
        results = []
        
        for x, y in queries:
            dist = abs(x) + abs(y)
            
            if len(max_heap) < k:
                # We need more elements in max_heap until it contains k elements
                heapq.heappush(max_heap, -dist)
            else:
                # If the new distance is smaller than the largest among the k smallest,
                # it should go into max_heap, and we move the largest of the k smallest into min_heap
                if dist < -max_heap[0]:
                    largest_smallest = -heapq.heappop(max_heap)
                    heapq.heappush(min_heap, largest_smallest)
                    heapq.heappush(max_heap, -dist)
                else:
                    # Otherwise, just push it to min_heap
                    heapq.heappush(min_heap, dist)
            
            # If we have fewer than k obstacles, result is -1
            if len(max_heap) < k:
                results.append(-1)
            else:
                # The k-th smallest distance is the root of max_heap (convert back from negative)
                results.append(-max_heap[0])
        
        return results