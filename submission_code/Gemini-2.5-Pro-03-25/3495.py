import heapq
from typing import List

class Solution:
  """
  Solves the problem of finding the k-th nearest obstacle after each query
  using a max-heap approach. The distance metric is Manhattan distance |x| + |y|.
  """
  def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
    """
    Processes queries to add obstacles and finds the k-th nearest distance after each.

    Args:
        queries: A list of [x, y] coordinates representing obstacle locations.
                 Each query adds a new, unique obstacle.
        k: The rank of the nearest obstacle distance to find (e.g., k=1 for nearest,
           k=2 for second nearest). Must be a positive integer.

    Returns:
        A list of integers where results[i] is the k-th nearest obstacle distance
        from the origin after query i has been processed. If fewer than k obstacles
        exist after query i, results[i] is -1.
    """
    # This list will store the result (k-th distance or -1) after each query.
    results = []
    
    # We use a min-heap data structure (provided by Python's heapq module)
    # to efficiently maintain the k smallest distances encountered so far.
    # To find the k-th smallest distance, we need the largest among these k smallest distances.
    # We simulate a max-heap using a min-heap by storing the *negative* of the distances (-d) 
    # in the min-heap `max_k_heap`.
    #
    # In this setup:
    # - The heap stores at most k elements (negative distances).
    # - The root `max_k_heap[0]` is the minimum element in the heap (the smallest negative value),
    #   which corresponds to the negative of the largest actual distance currently stored (-d_max).
    # - Therefore, `-max_k_heap[0]` gives us `d_max`. If the heap contains exactly k elements, 
    #   this `d_max` is the largest distance among the k smallest distances found, which is 
    #   precisely the k-th smallest distance.
    max_k_heap = [] 

    # Iterate through each query in the input list.
    for i, query in enumerate(queries):
        # Extract the x and y coordinates from the current query.
        x, y = query
        # Calculate the Manhattan distance of the new obstacle from the origin (0, 0).
        # Distance = |x - 0| + |y - 0| = |x| + |y|.
        distance = abs(x) + abs(y)

        # Now, update the heap to potentially include the new distance and maintain the 
        # property that it holds the k smallest distances seen so far.
        
        if len(max_k_heap) < k:
            # If the heap doesn't yet contain k elements, it means we have processed
            # fewer than k obstacles overall (or exactly k obstacles if this push makes size k). 
            # We simply add the negative of the new distance to the heap. 
            # `heapq.heappush` maintains the heap property.
            heapq.heappush(max_k_heap, -distance)
            
        elif distance < -max_k_heap[0]:
            # If the heap already contains k elements, we compare the new distance `distance`
            # with the largest distance currently in the heap (`d_max = -max_k_heap[0]`).
            # Note: `max_k_heap[0]` is the smallest element in the min-heap (most negative).
            # `-max_k_heap[0]` is the largest actual distance in the heap.
            #
            # If the new distance `distance` is strictly smaller than `d_max`,
            # it means this new obstacle is closer to the origin than the current k-th closest 
            # obstacle recorded in the heap.
            # Therefore, this new distance should be included in the set of k smallest,
            # and the old k-th distance (`d_max`) should be removed.
            #
            # `heapq.heapreplace(heap, item)` is the most efficient way to do this:
            # it first pops the smallest item from the heap (the root, `-d_max` in our case), 
            # then pushes the new item (`-distance` in our case), and finally restores the 
            # heap property. The size of the heap remains k.
            heapq.heapreplace(max_k_heap, -distance)
            
        # Otherwise (if len(max_k_heap) == k and distance >= -max_k_heap[0]):
        # The new distance is not smaller than the current k-th distance (the largest value 
        # currently tracked in the heap).
        # This means the new obstacle is not among the k closest obstacles found so far.
        # The set of k smallest distances does not change, so we do nothing to the heap.

        # After potentially updating the heap based on the current query, determine the 
        # result for this step (after processing query i).
        if len(max_k_heap) < k:
            # If, after processing the current query, the heap still has fewer than k elements,
            # it implies that the total number of obstacles added so far (i + 1) is less than k.
            # In this situation, the k-th distance is undefined. The problem asks for -1.
            results.append(-1)
        else:
            # If the heap has exactly k elements, it contains the negatives of the k smallest
            # distances among all obstacles added up to this query.
            # The k-th smallest distance is the largest value among these k distances.
            # As explained earlier, this value is obtained by negating the root of our
            # min-heap of negative distances: `-max_k_heap[0]`.
            kth_distance = -max_k_heap[0]
            results.append(kth_distance)

    # After processing all queries, return the list containing the result for each query step.
    return results