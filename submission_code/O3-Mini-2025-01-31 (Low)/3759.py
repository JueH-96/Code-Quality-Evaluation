from typing import List
import heapq

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        # Create a list of tuples (num1, num2, index)
        arr = [(nums1[i], nums2[i], i) for i in range(n)]
        # Sort by nums1
        arr.sort(key=lambda x: x[0])
        
        ans = [0] * n
        # Heap to maintain the k largest nums2 seen so far
        # We use a min-heap so that the smallest element among the top k is at the root.
        heap = []
        curr_sum = 0
        
        i = 0
        while i < n:
            # Process group with the same nums1 value
            j = i
            # For all elements in the current group, assign answer as current sum (from eligible previous elements)
            while j < n and arr[j][0] == arr[i][0]:
                _, _, orig_index = arr[j]
                ans[orig_index] = curr_sum
                j += 1
            
            # Now add all these group elements to our heap for future queries.
            for t in range(i, j):
                _, val2, _ = arr[t]
                # If we have less than k elements in our heap, add the element.
                if len(heap) < k:
                    heapq.heappush(heap, val2)
                    curr_sum += val2
                else:
                    # If this new value is larger than the smallest in the heap,
                    # then replace it to maintain the top k largest values.
                    if heap and heap[0] < val2:
                        curr_sum += val2 - heap[0]
                        heapq.heapreplace(heap, val2)
            i = j
        
        return ans