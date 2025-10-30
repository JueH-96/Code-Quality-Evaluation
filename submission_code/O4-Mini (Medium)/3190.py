from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # We'll try both possibilities at the last index: swap it or not.
        INF = 10**18
        best = INF
        a_last, b_last = nums1[n-1], nums2[n-1]
        
        # t = 0 means we do NOT swap at last index; t = 1 means we DO swap there.
        for t in (0, 1):
            # Determine the targets for the last position after t-swaps
            if t == 0:
                target1, target2 = a_last, b_last
            else:
                target1, target2 = b_last, a_last
            
            cost = t  # cost of the swap at the last index
            ok = True
            
            # For every other index i, decide if we must swap or not
            for i in range(n - 1):
                a_i, b_i = nums1[i], nums2[i]
                
                # If we do NOT swap at i, the pair is (a_i, b_i)
                no_swap_ok = (a_i <= target1 and b_i <= target2)
                # If we DO swap at i, the pair is (b_i, a_i)
                swap_ok    = (b_i <= target1 and a_i <= target2)
                
                if not no_swap_ok and not swap_ok:
                    # Impossible to make both newA[i] <= target1 and newB[i] <= target2
                    ok = False
                    break
                # If both options valid, pick no-swap to save operations
                if not no_swap_ok and swap_ok:
                    cost += 1
                # if no_swap_ok, we take it (cost +0), otherwise we already added cost for swap_ok
            if ok:
                best = min(best, cost)
        
        return best if best < INF else -1