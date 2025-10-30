from typing import List
import heapq

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        # Prepare a list of tuples with (value in nums1, value in nums2, original index)
        arr = [(nums1[i], nums2[i], i) for i in range(n)]
        # Sort by nums1's value in increasing order
        arr.sort(key=lambda x: x[0])
        
        # This will hold our final answers
        answer = [0] * n
        
        # min_heap is used to keep track of the top k values of nums2 from lower groups.
        # We also maintain a running sum of the values in the heap.
        min_heap = []
        curr_sum = 0
        
        # Process the sorted array in groups where nums1 values are equal.
        i = 0
        while i < n:
            curr_val = arr[i][0]
            # Find the group of indices with the same value in nums1.
            group = []
            while i < n and arr[i][0] == curr_val:
                group.append(arr[i])
                i += 1
            
            # For each element in this group, the valid j's come only from previous groups
            # (i.e. with a strictly smaller nums1 value), so answer is the current running sum.
            for _, _, orig_idx in group:
                answer[orig_idx] = curr_sum
            
            # Now update the heap with the nums2 values from this group so that they become
            # available for future queries (with higher nums1 values).
            for _, num2, _ in group:
                if len(min_heap) < k:
                    heapq.heappush(min_heap, num2)
                    curr_sum += num2
                else:
                    # If the heap is full, and the current num2 is larger than the smallest in heap,
                    # then replace it to maximize the overall sum.
                    if min_heap and num2 > min_heap[0]:
                        removed = heapq.heappop(min_heap)
                        curr_sum -= removed
                        heapq.heappush(min_heap, num2)
                        curr_sum += num2
        return answer