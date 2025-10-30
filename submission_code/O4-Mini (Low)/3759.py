from typing import List
import heapq

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        # Pair up nums1 value, nums2 value, and original index
        items = [(nums1[i], nums2[i], i) for i in range(n)]
        # Sort by nums1 value ascending
        items.sort(key=lambda x: x[0])
        
        # Min-heap to keep the top k nums2 values seen so far
        min_heap = []
        curr_sum = 0
        answer = [0] * n
        
        i = 0
        while i < n:
            # Find the range of items with the same nums1 value
            j = i
            while j < n and items[j][0] == items[i][0]:
                j += 1
            
            # For all items in [i, j), the heap currently contains
            # nums2 values of strictly smaller nums1, so record the sum
            for idx in range(i, j):
                _, _, orig_idx = items[idx]
                answer[orig_idx] = curr_sum
            
            # Now add the nums2 values of this group into the heap
            for idx in range(i, j):
                _, val2, _ = items[idx]
                heapq.heappush(min_heap, val2)
                curr_sum += val2
                # If we exceed k elements, remove the smallest
                if len(min_heap) > k:
                    removed = heapq.heappop(min_heap)
                    curr_sum -= removed
            
            # Move to the next group
            i = j
        
        return answer