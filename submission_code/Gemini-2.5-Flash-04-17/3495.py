from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        """
        Finds the distance of the k^th nearest obstacle from the origin after each query.

        Args:
            queries: A list of [x, y] coordinates where obstacles are placed.
            k: The rank of the nearest obstacle to find (k-th smallest distance).

        Returns:
            An integer array where results[i] is the k^th nearest obstacle distance
            after query i, or -1 if there are fewer than k obstacles.
        """
        results = []
        # Use a min-heap to store the k largest negative distances.
        # Storing negative distances and using a min-heap effectively simulates
        # a max-heap that stores the k smallest positive distances.
        # The smallest element in this min-heap (most negative) corresponds
        # to the largest positive distance among the k smallest.
        min_heap_neg_dist = []

        for x, y in queries:
            # Calculate the Manhattan distance from the origin
            dist = abs(x) + abs(y)

            # Push the negative distance onto the heap
            # If the heap size is less than k, just add it.
            # If the heap size is k, add it and then remove the largest (most negative) element.
            # This is equivalent to adding it and then popping if the new size > k.
            heapq.heappush(min_heap_neg_dist, -dist)

            # If the heap size exceeds k, remove the smallest element from the min-heap.
            # In a min-heap of negative numbers, the smallest element is the most negative one,
            # which corresponds to the largest positive distance. By removing it when size > k,
            # we ensure the heap keeps the k largest negative numbers, which represent
            # the k smallest positive distances seen so far.
            if len(min_heap_neg_dist) > k:
                heapq.heappop(min_heap_neg_dist)

            # After processing the current obstacle and ensuring the heap size is at most k,
            # check if we have accumulated at least k obstacles.
            # The number of obstacles is equal to the number of elements that have been
            # successfully placed in the heap. If the heap size is less than k, it means the total
            # number of obstacles encountered so far is less than k.
            if len(min_heap_neg_dist) < k:
                # Less than k obstacles so far
                results.append(-1)
            else:
                # We have exactly k elements in the heap, representing the k smallest distances.
                # The smallest element in this min-heap (min_heap_neg_dist[0]) is the k-th
                # largest negative number, which is the k-th smallest positive distance.
                results.append(-min_heap_neg_dist[0])

        return results