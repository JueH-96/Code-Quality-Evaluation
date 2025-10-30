import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Initialize is_marked array to keep track of marked elements.
        # False means unmarked, True means marked.
        is_marked = [False] * n
        
        # Calculate the initial sum of all elements.
        # This sum will be decremented as elements are marked.
        current_unmarked_sum = sum(nums)
        
        # Initialize a min-heap to efficiently find the smallest unmarked elements.
        # Each element in the heap is a tuple (value, original_index).
        # Python's heapq sorts tuples primarily by the first element (value),
        # then by the second element (index) in case of ties. This naturally
        # handles the tie-breaking rule: smallest value, then smallest index.
        min_heap = []
        for i in range(n):
            heapq.heappush(min_heap, (nums[i], i))
            
        # List to store the sum of unmarked elements after each query.
        answer = []
        
        # Process each query in order.
        for index_i, k_i in queries:
            # Step 1: Mark the element at index index_i.
            # Only mark if it's currently unmarked.
            if not is_marked[index_i]:
                is_marked[index_i] = True
                current_unmarked_sum -= nums[index_i]
            
            # Step 2: Mark k_i unmarked elements with the smallest values.
            # If multiple such elements exist, mark the ones with the smallest indices.
            # If fewer than k_i unmarked elements exist, then mark all of them.
            
            elements_marked_in_this_step = 0
            # Continue marking as long as we still need to mark more elements
            # (elements_marked_in_this_step < k_i) AND there are still elements
            # in the min-heap to consider.
            while elements_marked_in_this_step < k_i and min_heap:
                val, idx = heapq.heappop(min_heap)
                
                # Check if the element retrieved from the heap is currently unmarked.
                # An element popped from the heap might have been marked by:
                # 1. Step 1 of the current query.
                # 2. Step 1 of a previous query.
                # 3. Step 2 of a previous query.
                if not is_marked[idx]:
                    # If it's unmarked, mark it and update the sum.
                    is_marked[idx] = True
                    current_unmarked_sum -= val
                    elements_marked_in_this_step += 1
                # If it was already marked, we simply discard it from the heap
                # and continue to the next smallest element in the heap.
                # This ensures we only count truly unmarked elements towards the k_i count.
            
            # After both steps of the current query are processed,
            # record the current sum of all unmarked elements.
            answer.append(current_unmarked_sum)
            
        return answer