from typing import List
import heapq

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        # If k is 0, no selections can be made, so all answers are 0.
        if k == 0 or n == 0:
            return [0] * n
        
        # Pair up (value_from_nums1, value_from_nums2, original_index)
        arr = sorted((nums1[i], nums2[i], i) for i in range(n))
        
        # Min-heap to maintain the top k nums2 values seen so far
        heap = []
        curr_sum = 0
        answer = [0] * n
        
        i = 0
        while i < n:
            # Process a batch of equal nums1 values
            j = i
            # For all in this batch, the sum is based on strictly smaller nums1,
            # i.e., those already added to the heap.
            while j < n and arr[j][0] == arr[i][0]:
                _, val2, original_idx = arr[j]
                answer[original_idx] = curr_sum
                j += 1
            
            # Now add this batch's nums2 values into our top-k structure
            for p in range(i, j):
                _, val2, _ = arr[p]
                if len(heap) < k:
                    heapq.heappush(heap, val2)
                    curr_sum += val2
                else:
                    # If the new value is bigger than the smallest in the heap, replace
                    if heap and val2 > heap[0]:
                        smallest = heapq.heapreplace(heap, val2)
                        curr_sum += (val2 - smallest)
            i = j
        
        return answer