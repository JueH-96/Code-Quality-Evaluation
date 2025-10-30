import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # Initialize an empty heap and an empty set to store the coordinates of obstacles
        heap = []
        obstacles = set()
        # Initialize an empty list to store the results
        results = []
        # Iterate over the queries
        for x, y in queries:
            # If the coordinate is not in the set of obstacles, add it
            if (x, y) not in obstacles:
                obstacles.add((x, y))
                # Calculate the distance from the origin and add it to the heap
                distance = abs(x) + abs(y)
                heapq.heappush(heap, distance)
            # If the heap has more than k elements, remove the smallest
            if len(heap) > k:
                heapq.heappop(heap)
            # If the heap has k or more elements, add the kth nearest obstacle to the results
            if len(heap) == k:
                results.append(heap[0])
            # If there are less than k obstacles, add -1 to the results
            else:
                results.append(-1)
        # Return the results
        return results