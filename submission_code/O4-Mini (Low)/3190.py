from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        res = float('inf')
        
        # Try both choices for swapping at the last position
        for swap_last in (0, 1):
            # Determine the target maxima after deciding swap at n-1
            if swap_last == 0:
                t1, t2 = nums1[n-1], nums2[n-1]
            else:
                t1, t2 = nums2[n-1], nums1[n-1]
            
            ops = swap_last
            possible = True
            
            # For each i < n-1, pick whether to swap or not
            for i in range(n-1):
                a, b = nums1[i], nums2[i]
                # check if no-swap is valid
                can_no_swap = (a <= t1 and b <= t2)
                # check if swap is valid
                can_swap    = (b <= t1 and a <= t2)
                
                if not can_no_swap and not can_swap:
                    possible = False
                    break
                # If no-swap is possible, greedily take it (0 ops)
                if can_no_swap:
                    continue
                # Otherwise we must swap (1 op)
                ops += 1
            
            if possible:
                res = min(res, ops)
        
        return -1 if res == float('inf') else res