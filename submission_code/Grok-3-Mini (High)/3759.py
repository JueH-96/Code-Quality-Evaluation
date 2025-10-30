import bisect
import heapq
from typing import List

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        # Sort indices based on nums1
        sorted_idx = sorted(range(n), key=lambda i: nums1[i])
        # Values of nums2 in the order of sorted indices by nums1
        vals = [nums2[idx] for idx in sorted_idx]
        # Compute prefix sums of top k sums
        heap = []
        current_sum = 0
        sum_top_k = [0] * (n + 1)
        for m in range(1, n + 1):
            val = vals[m - 1]
            heapq.heappush(heap, val)
            current_sum += val
            if len(heap) > k:
                min_val = heapq.heappop(heap)
                current_sum -= min_val
            sum_top_k[m] = current_sum
        # Sorted nums1 for binary search
        sorted_nums1 = sorted(nums1)
        # Compute answer for each i
        answer = [0] * n
        for i in range(n):
            val = nums1[i]
            p = bisect.bisect_left(sorted_nums1, val)
            answer[i] = sum_top_k[p]
        return answer