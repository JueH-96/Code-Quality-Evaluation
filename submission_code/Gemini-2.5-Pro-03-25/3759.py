import heapq
from typing import List

class Solution:
  """
  Solves the problem of finding the maximum sum of k nums2 values 
  corresponding to nums1 values smaller than the current nums1 value for each index.
  """
  def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
    """
    Calculates the maximum sum for each index based on the problem specification.

    The approach involves sorting the elements based on nums1 values and then iterating
    through the sorted list. A min-heap is used to efficiently maintain the sum of the
    top k nums2 values encountered so far.

    Args:
      nums1: The first list of integers.
      nums2: The second list of integers.
      k: The maximum number of elements from nums2 to sum. Must be a positive integer.

    Returns:
      A list `answer` of size n, where answer[i] is the maximum sum calculated for the 
      element originally at index i.
    """
    n = len(nums1)

    # Step 1: Combine the data into tuples and store the original index.
    # Each tuple is (nums1_value, nums2_value, original_index).
    # This allows sorting based on nums1 while keeping track of the associated nums2 value
    # and the original position needed for the final answer array.
    pairs = []
    for i in range(n):
        pairs.append((nums1[i], nums2[i], i))

    # Step 2: Sort the combined data based on the nums1 values (the first element of the tuple).
    # Elements with the same nums1 value will be adjacent after sorting.
    # This ordering is crucial for the sliding window / cumulative approach.
    pairs.sort() 

    # Step 3: Initialize the result array.
    # It will store the calculated maximum sum for each original index.
    answer = [0] * n
    
    # Step 4: Initialize data structures for tracking the top k sums.
    # `top_k_heap`: A min-heap storing the k largest nums2 values encountered so far
    # among elements processed. Using a min-heap makes it easy to find and replace
    # the smallest element among the current top k.
    top_k_heap = [] 
    
    # `current_sum`: The sum of the elements currently present in `top_k_heap`.
    # This represents the sum of the top k (or fewer) nums2 values seen so far.
    current_sum = 0
    
    # Step 5: Iterate through the sorted pairs, processing them in groups based on nums1 values.
    i = 0 # Main pointer for iterating through sorted `pairs`.
    while i < n:
        # Identify the current group of elements having the same nums1 value.
        current_val1 = pairs[i][0]
        
        # Find the end index 'j' (exclusive) of this group in the sorted `pairs` list.
        j = i
        while j < n and pairs[j][0] == current_val1:
            j += 1

        # --- Phase A: Calculate and Assign Results for the Current Group ---
        # For every element within the current group (from index i to j-1),
        # the problem asks for the sum of the top k `nums2` values corresponding to
        # elements where `nums1` is strictly less than `current_val1`.
        # At this point, `current_sum` holds exactly this value, as it was computed
        # based on all elements processed *before* the current group.
        for batch_idx in range(i, j):
            # Retrieve the original index of the element being processed.
            original_index = pairs[batch_idx][2] 
            # Assign the calculated sum to the correct position in the `answer` array.
            answer[original_index] = current_sum

        # --- Phase B: Update Heap and Sum with Current Group's Values ---
        # Now, incorporate the `nums2` values from the current group (where `nums1 == current_val1`)
        # into the `top_k_heap` and update `current_sum`. These elements will contribute
        # to the calculations for subsequent groups (with larger `nums1` values).
        for batch_idx in range(i, j):
            # Get the nums2 value for the element in the current group.
            val2 = pairs[batch_idx][1] 

            # If the heap has fewer than k elements, we can simply add the current value.
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, val2)
                current_sum += val2 # Increment the total sum.
            # If the heap is already full (contains k elements), we check if the current `val2`
            # is larger than the smallest element currently in the top k (which is at the root
            # of the min-heap, `top_k_heap[0]`).
            elif val2 > top_k_heap[0]: 
                # If `val2` is larger, it deserves a place among the top k.
                # `heapreplace` efficiently removes the smallest element (`top_k_heap[0]`)
                # and inserts `val2`, maintaining the heap property. It returns the removed element.
                popped_val = heapq.heapreplace(top_k_heap, val2)
                # Adjust the `current_sum`: subtract the value that was removed, add the new value.
                current_sum = current_sum - popped_val + val2
                # If val2 <= top_k_heap[0], it's not among the top k, so we do nothing.
                
        # Move the main pointer 'i' to the start of the next group (index j).
        i = j

    # Step 6: Return the completed answer array.
    return answer