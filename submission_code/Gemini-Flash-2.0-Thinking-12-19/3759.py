import heapq
from typing import List

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        
        # 1. Create pairs (nums1[i], nums2[i], i)
        # We need the original index `i` to place the result in the `answer` array later.
        items = []
        for i in range(n):
            items.append((nums1[i], nums2[i], i))
            
        # 2. Sort items based on nums1 values.
        # Sorting by nums1 allows us to process indices in increasing order of their nums1 values.
        # This ensures that when considering an index i, all indices j with nums1[j] < nums1[i]
        # have already been processed and their corresponding nums2 values are available
        # as candidates for the top k sum.
        # If nums1 values are equal (a group), the relative order doesn't affect the set
        # of candidates for the top k sum *for this group*, as that set depends only on
        # values from *strictly smaller* nums1 groups.
        items.sort()
        
        # 3. Initialize answer array of size n with zeros.
        answer = [0] * n
        
        # 4. Initialize min-heap and current_sum.
        # The min-heap will store up to k elements, representing the k largest nums2 values
        # encountered so far from indices whose nums1 value is strictly less than the
        # nums1 value of the current group being processed.
        # The current_sum will maintain the sum of elements in the min-heap.
        min_heap = []
        current_sum = 0
        
        # Handle k=0 case explicitly: the sum is always 0.
        if k == 0:
            return answer # answer is already initialized with zeros

        # 6. Iterate through sorted items, processing groups with the same nums1 value together.
        i = 0
        while i < n:
            # 7a. Find the end index j of the current group.
            # All items from index i to j-1 inclusive have the same nums1 value (items[i][0]).
            j = i
            while j < n and items[j][0] == items[i][0]:
                j += 1
            
            # 7b. For each item in the current group (indices i to j-1), assign the current sum.
            # For any index `original_idx` in the current group, we need the sum of the top k
            # `nums2[p]` where `nums1[p] < nums1[original_idx]`. Since all items in the group
            # have the same `nums1` value (items[i][0]), the set of indices `p` where `nums1[p]`
            # is strictly less than this value is the same for all items in the group.
            # This set of indices `p` corresponds exactly to the items processed *before* index `i`
            # in the sorted list whose `nums1` value was strictly less than items[i][0].
            # The `current_sum` holds the sum of the top k nums2 values from these items.
            for group_item in items[i:j]:
                original_idx = group_item[2]
                answer[original_idx] = current_sum

            # 7c. Update heap and current_sum with values from the current group.
            # Add the nums2 values from the current group to the pool of candidates for the
            # top k sum calculation for *future* groups whose nums1 value is strictly greater
            # than the current group's nums1 value.
            for group_item in items[i:j]:
                v2 = group_item[1]
                
                # Maintain the min-heap of size at most k, containing the k largest values seen so far.
                # The heap should always store `k` elements if `k > 0` and more than `k` candidates
                # have been added.
                if len(min_heap) < k:
                    heapq.heappush(min_heap, v2)
                    current_sum += v2
                elif v2 > min_heap[0]:
                    # If the current value is larger than the smallest element in the heap (which is the
                    # smallest among the current top k), replace the smallest element with the current value.
                    smallest_v2 = heapq.heappop(min_heap)
                    current_sum -= smallest_v2
                    heapq.heappush(min_heap, v2)
                    current_sum += v2

            # 7d. Move the pointer i to the start of the next group.
            i = j
            
        return answer