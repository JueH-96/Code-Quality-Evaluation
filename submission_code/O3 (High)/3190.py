from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        INF = 10**9          # any large number works
        best = INF
        
        # Two possible pairs for the last position: keep it or swap it
        candidates = [(nums1[-1], nums2[-1], 0)]          # (max1 , max2 , cost for last index)
        if nums1[-1] != nums2[-1]:                        # the swap only makes sense if values differ
            candidates.append((nums2[-1], nums1[-1], 1))
        
        for max1, max2, last_cost in candidates:
            ops = last_cost
            possible = True
            
            for i in range(n - 1):
                a, b = nums1[i], nums2[i]
                
                # If current arrangement already respects the maxima, keep it
                if a <= max1 and b <= max2:
                    continue
                
                # Otherwise, try swapping this index
                if b <= max1 and a <= max2:
                    ops += 1
                else:              # neither keeping nor swapping works
                    possible = False
                    break
            
            if possible:
                best = min(best, ops)
        
        return -1 if best == INF else best