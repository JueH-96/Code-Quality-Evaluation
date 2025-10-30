import heapq
from typing import List

class Solution:
  def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
    n = len(grid)
    # Constraints: n >= 1. So grid is never empty.
    # m can be obtained from grid[0]
    m = len(grid[0]) 
    # Constraints: m >= 1. So grid[0] is never empty, and all rows have m elements.

    # Step 1: Pre-sort all rows in descending order.
    # sorted_rows_elements[i] will store the elements of grid[i] sorted descendingly.
    sorted_rows_elements = []
    for i in range(n):
        # sorted() returns a new list, original grid[i] is not modified.
        row_sorted_desc = sorted(grid[i], reverse=True)
        sorted_rows_elements.append(row_sorted_desc)

    # Step 2: Initialize a max-priority queue.
    # Python's heapq is a min-priority queue, so we store negated values.
    # Each item in PQ: (-value, row_index, element_index_in_sorted_row)
    # element_index_in_sorted_row is the index within sorted_rows_elements[row_index].
    pq = []

    # Step 3: Initial population of the priority queue.
    # For each row i, if its limit limits[i] allows taking at least one element,
    # and the row has elements (m > 0, guaranteed by constraints),
    # add its largest element (at index 0 of the sorted row) to the PQ.
    for i in range(n):
        # m > 0 means sorted_rows_elements[i] is not empty.
        if limits[i] > 0: 
            value = sorted_rows_elements[i][0]
            # Store (negative_value, row_index, element_index_in_this_row_sorted_list)
            heapq.heappush(pq, (-value, i, 0)) 

    # Step 4: Initialize total_sum.
    total_sum = 0

    # Step 5: Greedily pick elements.
    # Loop at most k times, or until the PQ becomes empty (no more valid elements to pick).
    for _ in range(k):
        if not pq: # If PQ is empty, no more elements can be picked.
            break
        
        # Pop the element with the current largest value.
        neg_val, r_idx, el_idx = heapq.heappop(pq)
        val = -neg_val
        total_sum += val
        
        # Try to add the next available element from the same row (r_idx) to the PQ.
        # The element just taken was sorted_rows_elements[r_idx][el_idx].
        # Its index in the sorted list for row r_idx was el_idx.
        # The next candidate element from this row would be at index (el_idx + 1).
        el_idx_next = el_idx + 1
        
        # Conditions to add s_r[el_idx_next] to PQ:
        # 1. The element s_r[el_idx_next] must exist: el_idx_next < m.
        # 2. The number of elements taken from row r_idx *including this next one*
        #    must not exceed limits[r_idx].
        #    The number of elements taken including s_r[el_idx_next] would be (el_idx_next + 1).
        #    So, (el_idx_next + 1) <= limits[r_idx], which is equivalent to el_idx_next < limits[r_idx].
        
        if el_idx_next < m and el_idx_next < limits[r_idx]:
            next_val = sorted_rows_elements[r_idx][el_idx_next]
            heapq.heappush(pq, (-next_val, r_idx, el_idx_next))
    
    # Step 6: Return the calculated maximum sum.
    return total_sum