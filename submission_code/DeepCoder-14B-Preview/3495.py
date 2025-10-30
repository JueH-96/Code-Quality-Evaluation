import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        small_heap = []  # max-heap implemented using min-heap with negative values
        large_heap = []  # min-heap
        results = []
        for x, y in queries:
            d = abs(x) + abs(y)
            if len(small_heap) < k:
                heapq.heappush(small_heap, -d)
            else:
                current_max_small = -small_heap[0]
                if d < current_max_small:
                    heapq.heappush(small_heap, -d)
                    # Extract the max from small_heap and add to large_heap
                    max_small = -heapq.heappop(small_heap)
                    heapq.heappush(large_heap, max_small)
                else:
                    heapq.heappush(large_heap, d)
            
            # Ensure all elements in large_heap are >= current_max_small (if small_heap is not empty)
            if small_heap:
                current_max_small = -small_heap[0]
                # Move elements from large_heap to small_heap if they are smaller than current_max_small
                while large_heap and large_heap[0] < current_max_small:
                    min_large = heapq.heappop(large_heap)
                    heapq.heappush(small_heap, -min_large)
                # Ensure small_heap does not exceed size k
                while len(small_heap) > k:
                    max_small = -heapq.heappop(small_heap)
                    heapq.heappush(large_heap, max_small)
            
            if len(small_heap) >= k:
                results.append(-small_heap[0])
            else:
                results.append(-1)
        return results