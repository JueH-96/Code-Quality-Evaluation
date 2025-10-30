import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        Calculates the sum of unmarked elements after a series of marking queries.

        Initially, all elements in the input array 'nums' are unmarked.
        Each query consists of two steps:
        1. Mark the element at the given index if it's not already marked.
        2. Mark k unmarked elements with the smallest values. If values are tied,
           mark the ones with the smallest indices. If fewer than k unmarked
           elements exist, mark all of them.

        Args:
            nums: A 0-indexed array of positive integers.
            queries: A 2D array of queries, where queries[i] = [index_i, k_i].

        Returns:
            An array of size m where answer[i] is the sum of unmarked elements
            after the i-th query.
        """
        n = len(nums)

        # Use a boolean array to track marked elements.
        # Initialized to False for all elements, meaning none are marked initially.
        marked = [False] * n

        # Initial sum of unmarked elements is the total sum of the array.
        # Python's built-in int type handles arbitrary precision, suitable for large sums.
        current_sum = sum(nums)

        # Create a min-heap of (value, index) pairs.
        # The heap automatically sorts elements first by value, then by index for ties
        # when comparing tuples. This is exactly what's needed for Step 2 of the query.
        min_heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(min_heap)

        # List to store the sum of unmarked elements after each query.
        answer = []

        # Process each query in the order they are given.
        for query in queries:
            index_i, k_i = query

            # Step 1: Mark the element at index_i if it is not already marked.
            # We only mark and update the sum if the element was previously unmarked.
            if not marked[index_i]:
                marked[index_i] = True
                current_sum -= nums[index_i]

            # Step 2: Mark k_i unmarked elements with the smallest values (and smallest indices in case of ties).
            # We need to find and mark up to k_i *unmarked* elements.
            # The heap provides the smallest elements efficiently. We iterate through the heap
            # until we mark k_i elements or the heap is empty (meaning no more unmarked elements
            # can be found via the heap).
            
            # Counter for elements successfully marked specifically in Step 2 for this query.
            marked_count_step2 = 0 
            
            # Loop while we still need to mark elements in this step (marked_count_step2 < k_i)
            # AND there are elements left in the heap to consider as potential candidates
            # for marking (min_heap is not empty).
            while marked_count_step2 < k_i and min_heap:
                # Get the smallest element (value, index) from the heap.
                # heapq.heappop() gives the element with the minimum (value, index) tuple.
                val, idx = heapq.heappop(min_heap)

                # Check if this element (at original index 'idx') is already marked.
                # An element might already be marked if it was handled by:
                # - A previous query's Step 1 or Step 2.
                # - The current query's Step 1 (if the popped index 'idx' happens to be equal to 'index_i').
                if not marked[idx]:
                    # If the element is currently unmarked, we mark it as part of Step 2.
                    marked[idx] = True
                    current_sum -= val
                    marked_count_step2 += 1
                # If marked[idx] is True, this element was already processed and marked in a previous step.
                # We do nothing more with this element for marking purposes; it's effectively
                # removed from consideration for future Step 2 markings once popped from the heap.
                # The loop continues to check the next smallest element from the heap.

            # After completing both Step 1 and Step 2 for the current query,
            # the 'current_sum' variable holds the sum of all elements that remain unmarked.
            # Append this sum to our answer list.
            answer.append(current_sum)

        # After processing all queries, return the list of sums.
        return answer