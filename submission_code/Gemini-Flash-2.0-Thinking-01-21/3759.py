import heapq
from typing import List

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        """
        Finds the maximum sum of at most k values from nums2[j] for indices j
        where nums1[j] < nums1[i], for each index i.

        Args:
            nums1: The first integer array.
            nums2: The second integer array.
            k: The maximum number of values to choose.

        Returns:
            An array where answer[i] is the result for index i.
        """
        n = len(nums1)
        
        # Create pairs (nums1[i], nums2[i], original_index)
        # This helps us sort based on nums1 while keeping track of the original position
        pairs = []
        for i in range(n):
            pairs.append((nums1[i], nums2[i], i))
            
        # Sort the pairs based on the nums1 values.
        # Processing elements in increasing order of nums1 values ensures that
        # when we consider nums1[i], all nums2[j] values for indices j
        # where nums1[j] < nums1[i] have been processed and added to the heap
        # (or considered for the heap).
        # If nums1 values are equal, the relative order does not impact the correctness
        # because all elements with the same nums1 value are processed together
        # *before* their nums2 values are considered for the heap.
        pairs.sort()
        
        # Initialize the result array with zeros
        answer = [0] * n
        
        # Initialize a min-heap to store the k largest nums2 values seen so far
        # from indices j where nums1[j] is strictly less than the current nums1[i].
        # Also maintain the sum of elements currently in the heap.
        min_heap = [] # Min-heap storing the k largest nums2 values (using a min-heap)
        current_sum = 0 # Sum of elements currently in the min_heap
        
        # Iterate through the sorted pairs. We process elements in blocks
        # that have the same nums1 value. This is important because for a given
        # nums1[i], we consider nums2[j] only if nums1[j] is *strictly* less.
        m = 0 # pointer to the start of the current block in sorted_pairs
        while m < n:
            # Get the nums1 value for the current block
            current_v1 = pairs[m][0]
            
            # Find the end index (exclusive) of the block containing the same nums1 value
            end_m = m
            while end_m < n and pairs[end_m][0] == current_v1:
                end_m += 1
                
            # Calculate the result for all original indices within the current block.
            # For an original index i where nums1[i] == current_v1, the eligible j indices
            # are those processed *before* this block (where nums1[j] < current_v1).
            # The `current_sum` at this moment is the sum of the top k nums2 values
            # from those preceding indices.
            for p in range(m, end_m):
                original_idx = pairs[p][2]
                answer[original_idx] = current_sum
            
            # After calculating results for the current block, update the heap
            # and sum with the nums2 values from this block. These values become
            # potential candidates for subsequent blocks (where nums1 values are strictly greater).
            # This update step happens AFTER the calculation step for the current block.
            for p in range(m, end_m):
                v2 = pairs[p][1]
                
                # Add the current nums2 value to the heap
                heapq.heappush(min_heap, v2)
                current_sum += v2
                
                # If the heap size exceeds k, remove the smallest element
                # to maintain the k largest elements.
                if len(min_heap) > k:
                    smallest = heapq.heappop(min_heap)
                    current_sum -= smallest
                    
            # Move the pointer to the start of the next block
            m = end_m
            
        return answer