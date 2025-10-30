import heapq
from typing import List

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        
        # Step 1: Pair nums1[i], nums2[i] with their original indices.
        # This tuple (nums1_val, nums2_val, original_index) allows us to:
        #   a) Sort by nums1_val.
        #   b) Retrieve the original index to place the result in the `answer` array.
        points = []
        for i in range(n):
            points.append((nums1[i], nums2[i], i))
        
        # Step 2: Sort the `points` list.
        # The primary sort key is `nums1_val` in ascending order.
        # When `nums1_val` are equal, the order of `nums2_val` or `original_index`
        # does not affect the correctness of the final results, as for these `i`
        # (where `nums1[i]` is the same), the set of eligible `j` (where `nums1[j] < nums1[i]`)
        # is identical. Python's default tuple sorting handles ties consistently.
        points.sort()
        
        # Step 3: Initialize the `answer` array with zeros.
        # This array will store the computed maximum sum for each original index `i`.
        answer = [0] * n
        
        # Step 4: Initialize a min-heap (priority queue).
        # This heap will store the `nums2` values of the `k` largest elements
        # that satisfy the `nums1[j] < nums1[i]` condition for the currently processed `i`.
        pq = [] # A list used as a min-heap by the `heapq` module.
        
        # Step 5: Initialize `current_sum`.
        # This variable maintains the sum of all elements currently in the `pq`.
        # This sum represents the maximum sum of at most `k` values from `nums2[j]`
        # where `nums1[j]` are strictly less than the `nums1` value of the current block being processed.
        current_sum = 0
        
        # Step 6: Iterate through the sorted `points` array.
        # We process points in blocks where `nums1` values are identical.
        i = 0 # `i` acts as a pointer to the start of the current block.
        while i < n:
            current_nums1_val = points[i][0]
            
            # Inner loop to process all points that have the same `nums1_val` as `current_nums1_val`.
            # For each such point `p`, its corresponding `nums1[p]` is equal to `current_nums1_val`.
            # Therefore, no previously processed `j` where `nums1[j] == current_nums1_val` is eligible.
            # All `j` currently in `pq` (and contributing to `current_sum`) satisfy `nums1[j] < current_nums1_val`.
            j = i # `j` acts as a pointer to iterate within the current block.
            while j < n and points[j][0] == current_nums1_val:
                # Assign the `current_sum` (calculated from elements with `nums1` values
                # strictly less than `current_nums1_val`) to the corresponding original index.
                answer[points[j][2]] = current_sum
                j += 1 # Move to the next point in the current block.
            
            # After calculating answers for all points in the current block (from `i` to `j-1`),
            # we now add their `nums2` values to the heap.
            # These `nums2` values will become candidates for subsequent blocks,
            # where `nums1` values will be strictly greater than `current_nums1_val`.
            # The loop goes from the start of the current block (`i`) up to `j` (exclusive).
            while i < j:
                val2_to_add = points[i][1]
                
                # Add the `nums2` value to the min-heap.
                heapq.heappush(pq, val2_to_add)
                current_sum += val2_to_add
                
                # If the heap size exceeds `k`, remove the smallest element.
                # This ensures `pq` always contains the `k` largest `nums2` values encountered so far.
                if len(pq) > k:
                    smallest_val_in_heap = heapq.heappop(pq)
                    current_sum -= smallest_val_in_heap
                
                i += 1 # Move `i` to the next element, advancing it past the current block.
                       # When this inner `while` loop finishes, `i` will be equal to `j`,
                       # correctly pointing to the start of the next distinct `nums1` block.
                       
        return answer