from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        a, b = nums1, nums2
        INF = float('inf')
        res = INF
        
        # We try both possibilities for the last index: swap it or not.
        for last_swap in (0, 1):
            if last_swap == 0:
                A_last = a[-1]
                B_last = b[-1]
                cost = 0
            else:
                A_last = b[-1]
                B_last = a[-1]
                cost = 1
            
            possible = True
            # For each i != n-1, decide swap or not to keep both arrays' values <= A_last/B_last
            for i in range(n - 1):
                # option 0: do not swap at i
                if a[i] <= A_last and b[i] <= B_last:
                    # best is not to swap (cost 0)
                    continue
                # option 1: swap at i
                if b[i] <= A_last and a[i] <= B_last:
                    cost += 1
                else:
                    possible = False
                    break
            
            if possible:
                res = min(res, cost)
        
        return -1 if res == INF else res