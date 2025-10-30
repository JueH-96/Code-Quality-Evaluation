import heapq
from typing import List

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        # Pair each value with its original index and sort by nums1 ascending
        sorted_pairs = sorted((v, i) for i, v in enumerate(nums1))
        
        # Heaps to maintain the top k nums2 values seen so far:
        # top_k is a min-heap of size at most k, containing the current top k values
        # rest is a max-heap (implemented with negative values) for all other values
        top_k = []          # min-heap
        rest = []           # max-heap via negatives
        sum_top = 0         # sum of elements in top_k
        
        answer = [0] * n
        p = 0
        # Process in groups of equal nums1 values
        while p < n:
            val = sorted_pairs[p][0]
            q = p
            # find range [p, q) with the same nums1 value
            while q < n and sorted_pairs[q][0] == val:
                q += 1
            
            # For all indices in [p, q), the available elements are exactly those
            # we've already inserted (which all have nums1 < val).
            for idx_in_sorted in range(p, q):
                _, original_idx = sorted_pairs[idx_in_sorted]
                answer[original_idx] = sum_top
            
            # Now insert these elements' nums2 values into our structure,
            # so that they count for future (larger nums1) queries.
            for idx_in_sorted in range(p, q):
                _, original_idx = sorted_pairs[idx_in_sorted]
                x = nums2[original_idx]
                
                if len(top_k) < k:
                    # Still room to grow our top_k heap
                    heapq.heappush(top_k, x)
                    sum_top += x
                else:
                    # If this x is bigger than the smallest in top_k, swap
                    if top_k and x > top_k[0]:
                        smallest_top = heapq.heappushpop(top_k, x)
                        sum_top += x - smallest_top
                        # put the displaced smallest into the rest
                        heapq.heappush(rest, -smallest_top)
                    else:
                        # Otherwise, it just goes into the rest
                        heapq.heappush(rest, -x)
            
            p = q
        
        return answer