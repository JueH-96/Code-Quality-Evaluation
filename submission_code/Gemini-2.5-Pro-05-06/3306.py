import heapq
from typing import List

class Solution:
  def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
    n = len(nums)
    
    # Prepare list for heapify
    # Each element in the heap will be a tuple (value, index)
    # This allows sorting by value, then by index for ties.
    heap_elements = []
    for i in range(n):
        heap_elements.append((nums[i], i))
    
    # Build the min-heap. heapq.heapify is O(N).
    heapq.heapify(heap_elements)
    # pq is an alias for heap_elements, used as the priority queue
    pq = heap_elements
        
    # Keep track of marked elements. Initially all are unmarked.
    marked = [False] * n
    
    # Calculate initial sum of all elements. sum() is O(N).
    # Since nums[i] are positive integers, sum can be large. Python integers handle this.
    current_sum = sum(nums) 
    
    answer = [] # To store the sum of unmarked elements after each query
    
    for index_i, k_i in queries:
        # Step 1: Mark the element at index_i if it is not already marked.
        # Check marked[index_i] to ensure we only process it if it's currently unmarked.
        if not marked[index_i]:
            marked[index_i] = True
            current_sum -= nums[index_i]
        
        # Step 2: Mark k_i unmarked elements in the array with the smallest values.
        # If multiple such elements exist, mark the ones with the smallest indices.
        # If less than k_i unmarked elements exist, mark all of them.
        
        elements_to_mark_in_this_step = k_i # Number of additional elements to mark
        
        # This loop continues as long as we still need to mark elements (elements_to_mark_in_this_step > 0)
        # AND there are still elements in the priority queue (pq is not empty).
        while elements_to_mark_in_this_step > 0 and pq:
            # Get the element with the smallest value (and smallest index for ties) from the heap.
            val, idx = heapq.heappop(pq)
            
            # Check if this element (nums[idx]) is already marked.
            # It could have been marked in Step 1 of this query, or in a previous query,
            # or in a previous iteration of this Step 2 loop.
            if not marked[idx]:
                # This is an unmarked element. Mark it.
                marked[idx] = True
                current_sum -= val # Subtract its value from the total sum of unmarked elements.
                elements_to_mark_in_this_step -= 1 # Decrement count of elements we still need to mark.
            # If marked[idx] is True:
            # This element is already marked. We are looking for k_i *unmarked* elements.
            # So, we skip this one. It does not count towards the k_i elements for this step.
            # The element is permanently removed from the heap by heappop.
            # The loop will continue to find the next smallest element from the heap.
        
        # After processing the query (both Step 1 and Step 2), 
        # store the current sum of unmarked elements.
        answer.append(current_sum)
            
    return answer