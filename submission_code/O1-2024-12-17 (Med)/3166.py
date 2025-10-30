class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        """
        We have an array nums of length n. We want to partition its indices into groups
        so that:
          1) Each group contains indices all having the same value in nums.
          2) The difference in sizes of any two groups is at most 1.
        We must return the minimum number of such groups.

        ------------------------------------------------------------------------
        KEY IDEA AND APPROACH:

        Let n = len(nums). Denote by c_1, c_2, ..., c_k the frequencies of the distinct
        values in nums (so sum of all c_i = n). We seek a grouping into g groups total,
        each group having size either s or s+1 for some integer s, with
            s = floor(n/g),
            r = n mod g,
        meaning exactly r groups have size s+1, and the other (g - r) groups have size s.

        Because each group can only hold indices of a single value, if c_i is to be split
        among (some) g_i groups, each of those g_i groups would be of size either s or s+1.
        That implies:

            c_i = s*g_i + x_i,

        where x_i is how many of those g_i groups are of size (s+1). Then 0 <= x_i <= g_i.
        Summing x_i across all i must total r, i.e. the total number of (s+1)-sized groups
        is r.  Equivalently:

            L*g_i <= c_i <= (L+1)*g_i,  where L = s = floor(n/g).

        And we also need sum_i g_i = g, because the total number of groups is g.

        One can show the following feasibility criterion for a fixed g:
          Let L = floor(n/g). Then for each frequency c_i,
              minG_i = ceil( c_i / (L+1) ),
              maxG_i = floor( c_i / L )       (if L>0)
          We need sum of minG_i <= g <= sum of maxG_i.
          If that holds, there is a valid way to split the c_i across exactly g groups,
          each of size L or L+1, and total groups = g.

        We want the minimal g satisfying this. The naive check would be to try g=1..n,
        but that can be O(n*k), which is too large (n and k can each be up to 1e5).

        ------------------------------------------------------------------------
        OPTIMIZATION (Enumerating floor(n/g) in descending order):

        Observe that floor(n/g) takes on only O(sqrt(n)) distinct values if g goes from 1..n.
        Indeed, when g changes, floor(n/g) remains constant over intervals of g, and there
        are about 2*sqrt(n) "breakpoints" in total. We can enumerate those distinct values
        of floor(n/g) – call each such value x. Then all g in the interval
            g in [ floor(n/(x+1)) + 1, floor(n/x ) ]
        yield the same x = floor(n/g).

        For each x, we compute:
           sumMin(x) = sum_i( ceil(c_i / (x+1)) )  = sum_i( (c_i + x)//(x+1) )
           sumMax(x) = sum_i( floor(c_i / x) )     = sum_i( c_i // x ),  if x>0
        Then the feasible g in that interval must satisfy:
           sumMin(x) <= g <= sumMax(x),  and
           g ∈ [ floor(n/(x+1)) + 1, floor(n/x) ].

        If there is an integer g in the intersection:
           [ sumMin(x), sumMax(x) ] ∩ [ floor(n/(x+1)) + 1, floor(n/x) ],
        then any such g is feasible. We want the smallest such g overall. Since smaller g
        corresponds to larger x (because x = floor(n/g)), we can iterate x in descending
        order and the first time we find a valid g, that must be the minimum feasible g.

        Edge cases:
          - If all values are the same (c_max = n), the answer is 1.
          - If x=0 or we get intervals out of range, we skip.
          - We always have a trivial upper bound of g = n (worst case each index alone).

        Complexity:
          - Building the frequency counts: O(n).
          - Collecting the O(sqrt(n)) distinct values of floor(n/g).
          - For each distinct x, we compute sumMin and sumMax in O(k), where k is the number
            of distinct values. In the worst case k = n = 1e5, sqrt(n) ~ 316 => 316 * 1e5 ~ 3.16e7,
            which can be borderline but often still doable in optimized Python or in C++.

        We'll implement this carefully and return the first valid g we find.
        ------------------------------------------------------------------------
        """

        import math
        from collections import Counter

        freq = Counter(nums)
        c = list(freq.values())
        n = len(nums)

        # Quick edge case: if there's only one distinct value or c_max == n
        if len(c) == 1:
            return 1
        c_max = max(c)
        if c_max == n:
            return 1

        # Helper to compute sumMin(x) = sum of ceil(c_i / (x+1))
        # which is sum( (c_i + x)//(x+1) ).
        def get_sum_min(x):
            s = 0
            denom = x + 1
            for val in c:
                s += (val + x) // denom
            return s

        # Helper to compute sumMax(x) = sum of floor(c_i / x), for x>0
        def get_sum_max(x):
            s = 0
            for val in c:
                s += val // x
            return s

        # Collect all distinct values of floor(n/g).
        # Standard trick to get them in O(sqrt(n)):
        distinct_x = set()
        limit = int(math.isqrt(n))  # floor(sqrt(n))
        for i in range(1, limit + 1):
            distinct_x.add(n // i)
            distinct_x.add(i)

        # We only care about x in [1..c_max], because if x > c_max, sumMax(x) = sum_i(c_i//x) = 0
        # (unless some c_i == x exactly), which generally won't help us form enough groups.
        # Also, x = 0 is invalid because floor(n/g)=0 => g>n, not possible.
        # So we'll clamp to [1.. c_max].
        distinct_x = [x for x in distinct_x if 1 <= x <= c_max]
        distinct_x.sort(reverse=True)  # Descending

        # We will iterate over x in descending order. For each x, the possible g range is
        #   g ∈ [ floor(n/(x+1))+1, floor(n/x) ]
        # Then check if there's a g in that range that also satisfies
        #   sumMin(x) <= g <= sumMax(x)
        # If yes, pick the smallest such g and return immediately.

        ans = None
        # A small helper to clamp an interval to [1..n].
        def clamp(lo, hi):
            return max(lo, 1), min(hi, n)

        prev_highg = 0  # (not strictly needed, but might be used if we wanted disjoint intervals)

        for x in distinct_x:
            # Interval for g
            lowg = (n // (x + 1)) + 1 if x + 1 != 0 else 1
            highg = n // x  # integer division

            # Clamp to [1..n]
            lowg, highg = clamp(lowg, highg)
            if lowg > highg:
                continue

            smin = get_sum_min(x)
            smax = get_sum_max(x) if x > 0 else 0  # x>0 guaranteed by our range

            Gmin = max(smin, lowg)
            Gmax = min(smax, highg)

            if Gmin <= Gmax:
                ans = Gmin
                break

        # If we never found a feasible g, then by problem statement we can always do g = n
        # (each index alone in its own group) as a fallback.
        if ans is None:
            ans = n

        return ans