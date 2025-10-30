import heapq
from typing import List

class Solution:
  def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
    results: List[int] = []
    
    # This list will be used as a min-heap by the heapq module.
    # To find the k-th smallest distance, we maintain a max-heap of size up to k
    # containing the k smallest distances seen so far. The top of this max-heap
    # (its largest element) would be the k-th smallest distance.
    # Since heapq implements a min-heap, we simulate a max-heap by storing
    # negative values. If D is a distance, we store -D in the min-heap.
    # The smallest element in this min-heap (heap[0]) corresponds to -D_kth,
    # where D_kth is the k-th smallest distance among those stored in the heap.
    # Thus, -heap[0] gives D_kth.
    k_smallest_distances_max_heap: List[int] = [] 

    for x_coord, y_coord in queries:
        # Calculate Manhattan distance from the origin
        distance = abs(x_coord) + abs(y_coord)

        if len(k_smallest_distances_max_heap) < k:
            # If the heap has fewer than k elements, we are still collecting the first k distances.
            # Add the new distance (negated for max-heap simulation).
            heapq.heappush(k_smallest_distances_max_heap, -distance)
        elif distance < -k_smallest_distances_max_heap[0]:
            # The heap is full (has k elements).
            # The current k-th smallest distance is -k_smallest_distances_max_heap[0].
            # If the new distance is smaller than this value, it means the new distance
            # should be in the set of k smallest, and the current largest among them
            # (which is -k_smallest_distances_max_heap[0]) should be removed.
            # heapq.heapreplace pops the root and pushes the new item, maintaining heap property.
            heapq.heapreplace(k_smallest_distances_max_heap, -distance)
        # Else (new_distance >= -k_smallest_distances_max_heap[0]):
        # The new distance is not smaller than the current k-th smallest.
        # So, it doesn't make it into the set of k smallest distances. The heap is unchanged.
            
        # After updating the heap, record the result for the current query.
        if len(k_smallest_distances_max_heap) < k:
            # If, after this query, we still have fewer than k obstacles in total,
            # then the k-th nearest obstacle doesn't exist yet.
            results.append(-1)
        else:
            # The heap has k elements, so -k_smallest_distances_max_heap[0] is the k-th smallest distance.
            results.append(-k_smallest_distances_max_heap[0])
    
    return results