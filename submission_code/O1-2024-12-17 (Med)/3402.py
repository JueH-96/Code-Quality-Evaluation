class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        """
        We want all elements of nums to be made equal by only *increasing* elements.
        We have two operations:
          1) Increment exactly 1 element by 1 at cost = cost1
          2) Increment exactly 2 distinct elements by 1 each at cost = cost2

        Return the minimum total cost (mod 1e9+7).

        ---------------------------------------------------------
        Key observations and approach:

        1) Let n = len(nums), mn = min(nums), mx = max(nums), S = sum(nums).

        2) We can only increase, so our final equal value v must be >= mx.
           However, if cost2 < 2*cost1, it might sometimes be cheaper overall
           to raise all elements to a value v > mx (to take advantage of many
           paired increments) rather than stopping at mx exactly.

        3) We define:
             T(v) = total increments needed = sum( (v - nums[i]) ) = n*v - S
             m(v) = the maximum single-element increments needed = v - mn
           Because the element that starts at the global minimum mn needs (v - mn).

        4) The maximum number of "double-increment" operations we can effectively use
           is limited by how many increments need to be done in total (T) and by the
           fact that we cannot pair an element if it no longer needs increments.
           A known formula is:
               number_of_pairs = min( floor(T/2), T - m )
           where (T - m) is the total increments needed by all elements *except*
           the one that needs the largest number of increments.  Intuitively,
           if one element needs more increments than the sum of the others,
           eventually it will be the *only* one that still needs increments,
           forcing single-increments at the end.

           Hence, once we fix v:
               p = min( T//2, T - m )    # number of double increments
               cost(v) = p*cost2 + (T - 2p)*cost1

           We must choose v >= mx to minimize cost(v).

        5) Cost function is piecewise depending on whether T >= 2*m or not:
             - If T >= 2*m, we can pair up almost everything, so cost(v) = floor(T/2)*cost2 + (T % 2)*cost1.
             - If T < 2*m, cost(v) = (T - m)*cost2 + (2*m - T)*cost1.

        6) We only really need to check a small set of candidate final values v:
             - v = mx   (the smallest feasible final value)
             - values around the "boundary" where T = 2*m, i.e. where n*v - S = 2*(v - mn).
               Solve (n*v - S) = 2(v - mn)  =>  (n-2)*v = S - 2*mn  =>  v0 = (S - 2*mn)/(n-2).
             Because for v < v0 we fall in the T<2m region, and for v >= v0 we fall in T>=2m region;
             in each region the cost is (essentially) linear or nearly linear, so the minimum
             will occur near that boundary or at the endpoints.

        7) Special cases:
             - If n <= 2, pairing doesn't really help if there's only 1 or 2 elements
               (for 1 element, cost=0; for 2 elements, only one might need increments,
               so effectively we just do single increments up to the larger).
             - If cost2 >= 2*cost1, it's never cheaper to do the double-increment operation
               than just 2 single-increments.  So we can simply raise everyone to mx
               by single-increments.

        8) Implementation steps:
             - Handle small n or cost2 >= 2*cost1 quickly.
             - Otherwise, compute v0.  Then gather a small set of candidate v:
                  { mx, mx+1, floor(v0)-1, floor(v0), floor(v0)+1, ... }
               restricted to >= mx, and check them.  Among those, pick the minimal cost.
             - Return result mod 1e9+7.

        We'll verify correctness using the given examples.
        """

        import math
        MOD = 10**9 + 7

        n = len(nums)
        if n == 0:
            return 0  # no cost if empty (though not in constraints)
        if n == 1:
            # Only one element => already equal
            return 0

        mn = min(nums)
        mx = max(nums)
        S = sum(nums)

        # If cost2 >= 2*cost1 => never beneficial to pair
        # Just raise all to mx with single increments
        if cost2 >= 2*cost1:
            return ((mx * n - S) * cost1) % MOD

        # If n == 2, pairing doesn't help if only one needs increments
        # So best is to raise the smaller to the bigger using single increments
        if n == 2:
            return ((mx - mn) * cost1) % MOD

        # Now n >= 3 and cost2 < 2*cost1
        # We'll check around the boundary v0 = (S - 2*mn)/(n-2).
        # We'll only consider integer v >= mx.
        # Because v must be at least mx (cannot go below max(nums)).

        def total_cost(v):
            # Compute cost for final value v
            T = n*v - S    # total increments
            m = v - mn     # largest increments needed by the min-element
            if T >= 2*m:
                # we can pair up almost everything
                pairs = T // 2
                leftover = T % 2
                return (pairs * cost2 + leftover * cost1)
            else:
                # limited by the largest one needing more than half the total
                return ((T - m) * cost2 + (2*m - T) * cost1)

        # Compute the boundary v0
        # (n-2)*v0 = (S - 2*mn) => v0 = (S - 2*mn)/(n-2), which might be float
        # We'll check the integer points around it, but not below mx.
        # It's possible (S - 2*mn) < 0 => then v0 < 0 => we'll clamp.
        v_candidates = set()
        v_candidates.add(mx)  # always check mx

        # Compute v0 if (n-2) != 0, but n>=3 => n-2>0 for n>2
        num = (S - 2*mn)
        den = (n - 2)

        # We'll consider floor and ceil around v0 if it is positive
        # but ensure v >= mx
        if den != 0:
            v0_float = num / den  # can be negative or positive
            for d in range(-2, 3):  # check a small neighborhood
                candidate = math.floor(v0_float) + d
                if candidate < mx:
                    continue
                # Only consider plausible final values
                if candidate >= mx:
                    v_candidates.add(candidate)

        # We'll limit how large a candidate can be. We only added at most ~5 or 6 around v0 anyway.
        # Filter out negative or zero final values (shouldn't happen if candidate >= mx>=1).
        v_candidates = {v for v in v_candidates if v >= mx}

        # Compute and return the minimum cost among candidates
        ans = None
        for v in v_candidates:
            c = total_cost(v)
            if ans is None or c < ans:
                ans = c

        return ans % MOD