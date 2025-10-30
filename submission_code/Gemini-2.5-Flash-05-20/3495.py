import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        """
        Calculates the k-th nearest obstacle distance from the origin after each query.

        Args:
            queries: A list of [x, y] coordinates for obstacles.
            k: The k-th nearest obstacle to find.

        Returns:
            An integer array where results[i] denotes the k-th nearest obstacle distance
            after query i, or -1 if there are less than k obstacles.
        """
        results = []
        
        # max_heap is a min-heap in Python that stores negative distances.
        # This setup effectively simulates a max-heap.
        # The element at max_heap[0] will be the smallest negative value,
        # which corresponds to the largest positive distance among the 'k' elements in the heap.
        # This largest positive distance in the heap represents the k-th smallest
        # overall distance encountered so far.
        max_heap = [] 

        for x, y in queries:
            distance = abs(x) + abs(y)

            # If the heap contains fewer than 'k' elements, simply add the new distance.
            # We push the negative of the distance to simulate a max-heap with heapq (min-heap).
            if len(max_heap) < k:
                heapq.heappush(max_heap, -distance)
            else:
                # If the heap already contains 'k' elements, we compare the new distance
                # with the current k-th smallest distance (which is -max_heap[0]).
                if distance < -max_heap[0]:
                    # If the new distance is smaller, it means it should be one of the
                    # 'k' smallest. We remove the current largest element from the heap
                    # (which is the current k-th smallest overall) and add the new, smaller one.
                    heapq.heappop(max_heap)  # Remove the current largest element from the simulated max-heap
                    heapq.heappush(max_heap, -distance) # Add the new, smaller distance

            # After processing the current query and updating the heap,
            # determine the k-th smallest distance to record in results.
            if len(max_heap) < k:
                # If we still have fewer than 'k' obstacles, the k-th obstacle doesn't exist yet.
                results.append(-1)
            else:
                # The k-th smallest distance is the largest value in our simulated max-heap,
                # which is represented by -max_heap[0].
                results.append(-max_heap[0]) 
                
        return results