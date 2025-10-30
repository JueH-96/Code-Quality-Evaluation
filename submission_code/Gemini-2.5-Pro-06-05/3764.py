import heapq
from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        """
        Finds the maximum sum of at most k elements from the matrix grid,
        respecting the per-row limits.
        """
        n = len(grid)
        m = len(grid[0])

        # Step 1: Sort each row in descending order. This allows us to consider
        # elements in a greedy fashion (from largest to smallest) for each row.
        for i in range(n):
            grid[i].sort(reverse=True)

        # Step 2: Initialize a max-heap to keep track of the largest available element
        # from each row. Python's `heapq` is a min-heap, so we store negative
        # values to simulate a max-heap.
        # The heap stores tuples: (-value, row_index, column_index).
        max_heap = []
        for i in range(n):
            # If a row has a non-zero limit and contains elements,
            # push its largest element onto the heap. (m >= 1 from constraints).
            if limits[i] > 0:
                heapq.heappush(max_heap, (-grid[i][0], i, 0))

        # Step 3: Greedily extract the largest element from the heap up to `k` times.
        total_sum = 0
        for _ in range(k):
            # If the heap is empty, we have exhausted all possible elements
            # within the given limits.
            if not max_heap:
                break

            # Pop the globally largest available element.
            neg_val, row_idx, col_idx = heapq.heappop(max_heap)
            
            # Add its value to the total sum.
            total_sum += -neg_val

            # Step 4: Add the next element from the same row to the heap if valid.
            next_col_idx = col_idx + 1
            
            # We can push the next element if:
            # a) It exists in the row (index is within the row's bounds).
            # b) The number of elements taken from this row is still under its limit.
            #    We have taken `col_idx + 1` elements so far. The next one is valid
            #    if `col_idx + 1 < limits[row_idx]`.
            if next_col_idx < m and next_col_idx < limits[row_idx]:
                heapq.heappush(max_heap, (-grid[row_idx][next_col_idx], row_idx, next_col_idx))
        
        return total_sum