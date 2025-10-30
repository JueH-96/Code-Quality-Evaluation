from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        nums_min = min(nums)
        nums_max = max(nums)
        total = sum(nums)
        
        # cost to reach H: helper
        def cost_for_H(H: int) -> int:
            # total increments needed
            S = n * H - total
            if S <= 0:
                return 0
            # if single ops always no worse
            if cost2 >= 2 * cost1 or n <= 2:
                return S * cost1
            # otherwise we can use pair ops
            d_max = H - nums_min
            # max pairs we can do
            # if d_max <= S/2, we can do floor(S/2) pairs
            if d_max * 2 <= S:
                pairs = S // 2
            else:
                # otherwise limited by sum - d_max
                pairs = S - d_max
            singles = S - 2 * pairs
            return pairs * cost2 + singles * cost1
        
        # If n <= 2 or pairing never helps, best is at H = nums_max
        if n <= 2 or cost2 >= 2 * cost1:
            ans = cost_for_H(nums_max) % MOD
            return ans
        
        # For n >= 3 and cost2 < 2*cost1, we check possible H candidates.
        # Compute slope in regime B: s2 = (n-1)*cost2 - (n-2)*cost1
        s2 = (n - 1) * cost2 - (n - 2) * cost1
        
        # Compute transition H where regime B ends: H_t = (total - 2*nums_min) / (n-2)
        nume = total - 2 * nums_min
        den = n - 2
        
        # Collect candidate H's
        candidates = {nums_max}
        
        # If regime B slope is negative, best in B is at the transition boundary
        if s2 < 0:
            # compute floor and ceil of H_t
            # be careful with possible negative numerator
            # floor
            Hf = nume // den
            # ceil
            # ceil(n/d) = (n + d - 1) // d when d > 0
            Hc = (nume + den - 1) // den
            for h in (Hf, Hc):
                if h >= nums_max:
                    candidates.add(h)
        
        # Evaluate costs and pick minimum
        best = None
        for H in candidates:
            c = cost_for_H(H)
            if best is None or c < best:
                best = c
        
        return best % MOD