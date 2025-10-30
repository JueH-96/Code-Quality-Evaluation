import heapq
from typing import List

class Solution:
  """
  Solves the problem of finding the sum of unmarked elements in an array after a series of marking queries.
  Queries involve marking a specific index and then marking k smallest unmarked elements, prioritizing smaller indices in case of ties.
  """
  def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
    """
    Calculates the sum of unmarked elements after each query.

    Args:
      nums: The list of positive integers.
      queries: A list of queries, where each query is [index_i, k_i].
               index_i is the index to mark initially.
               k_i is the number of smallest unmarked elements to mark subsequently.

    Returns:
      A list where the i-th element is the sum of unmarked elements after the i-th query.
    """
    n = len(nums)
    
    # Keep track of marked elements using a boolean array for O(1) access.
    # Initially, all elements are unmarked.
    marked = [False] * n
    
    # Calculate the initial total sum of all elements in the array.
    # We will subtract from this sum as elements get marked.
    # Using sum() is generally efficient for this.
    total_sum = sum(nums) 
    
    # Prepare items to be placed in the min-heap.
    # Each item is a tuple (value, index). Storing the index is crucial for two reasons:
    # 1. Tie-breaking: If two elements have the same value, the heap (using tuple comparison)
    #    will prioritize the one with the smaller index, as required by the problem.
    # 2. Tracking: We need the index to update the `marked` array correctly.
    heap_items = []
    for i in range(n):
        heap_items.append((nums[i], i))
        
    # Create a min-heap from the list of (value, index) tuples.
    # heapq.heapify builds the heap in O(n) time.
    # This heap allows us to efficiently find the unmarked element with the smallest value
    # (and smallest index in case of ties).
    min_heap = heap_items 
    heapq.heapify(min_heap) 
    
    # List to store the result (sum of unmarked elements) after each query.
    answer = []
    
    # Process each query sequentially in the order they appear in the `queries` list.
    for index_i, k_i in queries:
        
        # --- Step 1: Mark the element at the specified index_i ---
        # Check if the element at index_i is not already marked before marking it.
        if not marked[index_i]:
            # Mark the element by setting its corresponding flag to True.
            marked[index_i] = True
            # Subtract its value from the running total sum of unmarked elements.
            total_sum -= nums[index_i]
            # Note on Heap Management: We use "lazy deletion". The element (nums[index_i], index_i)
            # might still exist within the `min_heap`. However, because `marked[index_i]`
            # is now True, if we encounter this element when extracting from the heap later,
            # we will simply discard it (see Step 2).

        # --- Step 2: Mark k_i additional unmarked elements with the smallest values ---
        marked_count = 0 # Initialize a counter for elements marked in this step.
        
        # Continue marking the smallest unmarked elements as long as:
        # 1. We haven't yet marked k_i elements in this step (`marked_count < k_i`).
        # 2. There are still potential candidates in the heap (`min_heap` is not empty).
        while marked_count < k_i and min_heap:
            
            # --- Lazy Deletion from Heap Top ---
            # Before extracting the minimum element, we must ensure it hasn't been marked already
            # (either in Step 1 of this query, earlier in Step 2, or by a previous query).
            # Check the element at the top of the heap (`min_heap[0]`).
            # If the index associated with the top element (`min_heap[0][1]`) corresponds
            # to an element that is already marked (`marked[min_heap[0][1]]` is True),
            # then this element is outdated (already marked). Pop it from the heap
            # and repeat the check with the new top element.
            while min_heap and marked[min_heap[0][1]]:
                heapq.heappop(min_heap) # Discard the already marked element.
                
            # If the heap becomes empty after removing all marked elements from its top,
            # it means there are no more unmarked elements left in the array.
            # We cannot mark any more elements, so break out of the loop.
            if not min_heap:
                break
                
            # --- Extract and Mark the Smallest Unmarked Element ---
            # At this point, the element at the top of the heap (`min_heap[0]`) is guaranteed
            # to be the (value, index) pair corresponding to the currently smallest unmarked
            # element (considering the index for tie-breaking).
            # Extract this smallest element using heappop.
            value, index = heapq.heappop(min_heap)
            
            # Mark this element as visited by setting its flag in the `marked` array.
            # We know it was unmarked when popped due to the lazy deletion loop above.
            marked[index] = True
            # Subtract its value from the total sum of unmarked elements.
            total_sum -= value
            # Increment the counter for elements marked in this specific step (Step 2).
            marked_count += 1

        # --- Store Result for the Current Query ---
        # After completing both steps (marking index_i and up to k_i smallest),
        # the value of `total_sum` correctly reflects the sum of all elements
        # that are still unmarked. Append this sum to the `answer` list.
        answer.append(total_sum)
        
    # After processing all queries, return the list containing the sum
    # of unmarked elements recorded after each query.
    return answer