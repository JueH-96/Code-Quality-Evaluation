from typing import List
import heapq

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        # indices sorted by nums1 value
        order = list(range(n))
        order.sort(key=lambda i: nums1[i])

        answer = [0] * n

        # min-heap that stores the k largest nums2 values seen so far
        heap: List[int] = []
        current_sum = 0          # sum of elements currently in the heap

        i = 0
        while i < n:
            val = nums1[order[i]]

            # find the whole block with the same nums1 value
            j = i
            while j < n and nums1[order[j]] == val:
                j += 1

            # assign answers for this block (only elements with smaller nums1 counted)
            for p in range(i, j):
                answer[order[p]] = current_sum

            # now insert this block's nums2 values for future (larger) nums1 values
            for p in range(i, j):
                v2 = nums2[order[p]]
                heapq.heappush(heap, v2)
                current_sum += v2
                if len(heap) > k:               # keep only the k largest values
                    current_sum -= heapq.heappop(heap)

            i = j

        return answer