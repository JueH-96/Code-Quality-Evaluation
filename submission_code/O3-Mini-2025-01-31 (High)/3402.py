from math import ceil
from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # If only one element, no cost.
        if n == 1:
            return 0

        s = sum(nums)
        m = max(nums)
        a = min(nums)
        
        # Given a final target T, we must add (T - nums[i]) to each entry.
        # Let S = total increments = n * T - s.
        # However, we are allowed to combine increments (on two different indices) for cost cost2,
        # but such pair operations can only be used as long as you can “match” increments – the maximum
        # number of pair operations is limited by:
        #    P_max = min( S//2, S - (T - a) )
        # because the biggest gap is T - a (for the minimum number).
        # Then the cost for target T is:
        #    f(T) = P * cost2 + (S - 2*P) * cost1,
        # where P = min(S//2, S - (T - a)).
        def cost_for_T(T: int) -> int:
            S = n * T - s
            # When S is 0 no operations are needed.
            if S <= 0:
                return 0
            p1 = S // 2
            # The element with the minimum value needs T - a increments.
            # Since it cannot be paired with itself, the total pairable increments are limited by S - (T - a)
            p2 = (n - 1) * T - (s - a)
            # p2 might be negative if T is too low; however if T >= m then (T - a) is at most (m - a) for at least one index (the max)
            # so p2 will be >= 0.
            P = min(p1, p2)
            # After using P pair operations, the remaining unpaired increments are S - 2*P, done one‐by‐one.
            return P * cost2 + (S - 2 * P) * cost1

        # If the cost for single-index increment is cheaper (or as cheap) as doing a pair,
        # then it never makes sense to make a “balanced” raise that forces pairing.
        # (Because a pair operation costs cost2 but doing two singles costs 2*cost1.)
        if cost2 >= 2 * cost1:
            ans = (n * m - s) * cost1
            return ans % MOD

        # For n==2, one of the numbers is already = m, so there is no opportunity to pair.
        if n == 2:
            ans = cost_for_T(m)
            return ans % MOD

        # Now n >= 3 and cost2 < 2*cost1.
        # We are free to choose a final target T such that T >= m.
        # Note: raising T above m forces every element to be increased.
        # That may allow us to pair operations much more efficiently.
        # In particular, if you set T = m, then only those elements below m need increments.
        # In contrast, if T is higher, then even the elements already at m must be increased – 
        # but then many indices may be “active” so that many increments can be paired.
        #
        # Our cost function for a given T is:
        #    f(T) = P * cost2 + (n * T - s - 2P) * cost1, where P = min( (n*T - s)//2, (n-1)*T - (s - a) )
        #
        # A useful observation is that the maximum pairing count is limited
        # by the imbalance: P_max = S - (T - a) = (n - 1)*T - (s - a).
        # And the other natural limit is S//2.
        # The breakpoint is when
        #    S//2   ==   (n-1)*T - (s - a)
        # (ignoring floors) this is roughly when:
        #    (n*T - s) / 2 = (n-1)*T - (s - a)
        # Rearranging (and treating the equality continuously) gives:
        #    T*(n-2) = s - 2a.
        # So define X = (s - 2a) / (n-2) and let T0 = ceil(X).
        # For T < T0, the distribution is “unbalanced” so that the maximum pairs we can do is limited by (n-1)*T - (s-a).
        # For T >= T0, the limit is S//2.
        #
        # Because our function is piecewise‐linear and the optimum always occurs at a boundary point,
        # it is enough to check a few candidate integer values in the vicinity of m and T0.
        #
        # We'll always consider T = m, m+1, m+2.
        # And if T0 is at least m, we also check T0 - 1, T0, T0 + 1, T0 + 2.
        candidates = set()
        # Always valid candidates
        candidates.add(m)
        candidates.add(m + 1)
        candidates.add(m + 2)
        
        # Compute the breakpoint T0 from the equation T*(n-2) = s - 2a.
        # For n>=3 the denominator is positive.
        T0 = ceil((s - 2 * a) / (n - 2))
        # Only consider these candidates if T0 is at least m.
        if T0 >= m:
            for delta in [-1, 0, 1, 2]:
                T_candidate = T0 + delta
                if T_candidate >= m:
                    candidates.add(T_candidate)
        
        # Evaluate cost_for_T for each candidate and take the minimum.
        best = None
        for T_candidate in candidates:
            current = cost_for_T(T_candidate)
            if best is None or current < best:
                best = current
        
        return best % MOD