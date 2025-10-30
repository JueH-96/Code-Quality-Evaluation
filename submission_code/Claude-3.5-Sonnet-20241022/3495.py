class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # Store distances in a sorted list
        distances = []
        result = []
        
        for x, y in queries:
            # Calculate Manhattan distance
            dist = abs(x) + abs(y)
            
            # Insert distance in sorted order
            left, right = 0, len(distances)
            while left < right:
                mid = (left + right) // 2
                if distances[mid] > dist:
                    right = mid
                else:
                    left = mid + 1
            distances.insert(left, dist)
            
            # Check if we have k obstacles yet
            if len(distances) < k:
                result.append(-1)
            else:
                result.append(distances[k-1])
                
        return result