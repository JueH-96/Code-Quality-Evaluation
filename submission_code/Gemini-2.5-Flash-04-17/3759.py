import heapq
from typing import List

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        
        # Create pairs (nums1[i], nums2[i], original_index)
        indexed_pairs = [(nums1[i], nums2[i], i) for i in range(n)]
        
        # Sort pairs based on nums1 values.
        # If nums1 values are equal, the relative order doesn't strictly matter
        # for correctness, as all indices with the same nums1 value will rely
        # on the same set of nums2 values from indices with strictly smaller
        # nums1 values. Python's sort is stable.
        indexed_pairs.sort()
        
        # Initialize the answer array of size n
        answer = [0] * n
        
        # Use a min-heap to efficiently maintain the k largest nums2 values.
        # The heap will store nums2 values encountered so far from indices j
        # where nums1[j] is strictly less than the current nums1 value being processed.
        min_heap = []
        
        # Keep track of the sum of elements currently in the min-heap.
        current_sum = 0
        
        # Iterate through the sorted pairs, processing indices in groups
        # where nums1 values are the same.
        start = 0
        while start < n:
            # Find the end index of the current group with the same nums1 value.
            current_v = indexed_pairs[start][0]
            end = start
            while end < n and indexed_pairs[end][0] == current_v:
                end += 1
            
            # For each index i within the current group (from start to end-1):
            # The set of indices j where nums1[j] < nums1[i] corresponds exactly
            # to the indices represented by the elements processed *before* this group
            # in the sorted list.
            # Therefore, the sum of the top k nums2 values for nums1[i] is
            # the 'current_sum' accumulated from those previous elements.
            # The heap state *before* adding values from the current group
            # correctly reflects this set of 'previous' indices.
            for i in range(start, end):
                original_idx = indexed_pairs[i][2]
                answer[original_idx] = current_sum
            
            # After calculating results for the current group, we add the nums2 values
            # from the indices in this group to our consideration set (the min-heap).
            # These values will potentially be included in the sum calculation for
            # subsequent groups (where nums1 values are strictly greater).
            for i in range(start, end):
                w = indexed_pairs[i][1]
                
                # Add the current nums2 value to the heap and update the running sum.
                heapq.heappush(min_heap, w)
                current_sum += w
                
                # If the heap size exceeds k, remove the smallest element.
                # This ensures the heap always contains at most k elements,
                # which are the k largest values seen so far.
                if len(min_heap) > k:
                    min_w = heapq.heappop(min_heap)
                    current_sum -= min_w
            
            # Move the start pointer to the beginning of the next group.
            start = end
        
        return answer