import bisect

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        distances = []
        results = []
        
        for query in queries:
            x, y = query
            distance = abs(x) + abs(y)
            
            # Insert the new distance into the sorted list
            bisect.insort(distances, distance)
            
            # Determine the k-th nearest obstacle
            if len(distances) < k:
                results.append(-1)
            else:
                # The k-th smallest distance
                results.append(distances[k - 1])
        
        return results