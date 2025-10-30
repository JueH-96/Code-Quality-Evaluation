class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        import heapq
        
        results = []
        k_smallest = []  # Min-heap of negated distances to keep the k smallest distances
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            
            if len(k_smallest) < k:
                heapq.heappush(k_smallest, -distance)
            elif -distance > k_smallest[0]:
                heapq.heapreplace(k_smallest, -distance)
            
            if len(k_smallest) < k:
                results.append(-1)  # Less than k obstacles
            else:
                results.append(-k_smallest[0])  # k-th nearest obstacle
        
        return results