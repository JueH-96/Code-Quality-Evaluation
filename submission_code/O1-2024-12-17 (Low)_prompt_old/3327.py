class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        """
        We use the following key observations to derive an O(n) solution:

        1) Placing a new '1' uses one "change" (if we have not exceeded maxChanges) and costs 2 moves
           in total to pick it up (1 move to place, 1 move to swap it into aliceIndex).
        2) Moving an existing '1' from distance d>0 into aliceIndex costs d moves (one swap per unit distance).
           - If d=1, cost=1, which is cheaper than creating a new '1' (which costs 2).
           - If d=2, cost=2, which is the same as creating a new '1'.
           - If d>2, cost>d, which is strictly more expensive than creating a '1' (which costs 2).
        3) Therefore, for an index c, only the existing '1's at distance <= 2 from c can be relevant at cost<=2 each.
           Any '1' farther than distance=2 is never cheaper than simply creating a new '1'.
        4) We also get a "free" pick-up if nums[c]==1 at the start (distance=0). That reduces the needed count k by 1.
        5) Hence, to see if picking from index c is feasible (and at what cost):
           - Let freeOne = 1 if nums[c]==1 else 0.
           - We then need (k - freeOne) more '1's.
           - Let cDist0 = freeOne
             cDist1 = number of '1's within distance <=1 of c
             cDist2 = number of '1's within distance <=2 of c
             (excluding those counted in cDist1).
           - We can use up to cDist0 + cDist1 + cDist2 existing '1's at cost <= 2 each,
             plus we can create up to maxChanges new '1's (also cost=2 each).
           - Therefore, we must have (k) <= (cDist0 + cDist1 + cDist2 + maxChanges) to be able to pick up k ones.
           - The cost calculation then goes as follows:
               * We automatically take the freeOne if cDist0=1 (this costs 0 moves for that one).
               * Then we greedily pick as many distance=1 '1's as we still need, since each of those costs 1 move
                 (strictly cheaper than 2). Suppose we pick used1 = min(cDist1 - cDist0, k - cDist0).
                 (This is how many "distance=1" ones we actually use, excluding the freeOne if any.)
               * The leftover needed = (k - cDist0) - used1.  Each leftover one costs 2 moves (whether from distance=2 or
                 from newly created ones). Thus the cost from leftover is 2 * leftover.
               * The cost from those distance=1 picks is exactly used1 (one move each).
               * Total = used1 + 2 * leftover = 2*(k - cDist0) - used1
                 (since leftover = (k - cDist0) - used1, so used1 + 2*leftover = 2*(k - cDist0) - used1).
           - We check feasibility by ensuring leftover <= (cDist2 + maxChanges). If leftover is larger, we cannot
             supply that many "2-move picks" from existing distance=2 ones plus new ones, hence infeasible.

        Algorithm in steps:
        - Build a prefix sum array ps so that we can quickly count how many '1's are in any interval [L, R].
        - For each c in [0..n-1]:
            cDist0 = 1 if nums[c]==1 else 0
            cDist1 = number of '1's in [c-1..c+1] (clip to array bounds)
            cDist2Full = number of '1's in [c-2..c+2] (clip to array bounds)
            cDist2 = cDist2Full - cDist1
            needed = k - cDist0
            if needed <= 0:
                # means k=1 and we started on a '1', or k=0 corner case => cost=0
                cost=0
            else:
                c1Excl = cDist1 - cDist0  # excludes the freeOne if cDist0=1
                used1 = min(c1Excl, needed)
                leftover = needed - used1
                # Check feasibility:
                if leftover > cDist2 + maxChanges:  # can't fill leftover from distance=2 plus new ones
                    cost = large_number  # infeasible
                else:
                    cost = 2*needed - used1
            take the minimum cost over all c
        - Return that minimum cost.

        This runs in O(n) time (after O(n) preprocessing for prefix sums).
        """

        import sys
        INF = 10**15

        n = len(nums)
        # Build prefix sum array for quick range queries
        ps = [0]*(n+1)
        for i in range(n):
            ps[i+1] = ps[i] + nums[i]

        def count_ones_in_range(L, R):
            # clip to [0, n-1]
            if R < 0 or L > n-1: 
                return 0
            L = max(L, 0)
            R = min(R, n-1)
            if L>R:
                return 0
            return ps[R+1] - ps[L]

        best = INF

        for c in range(n):
            freeOne = nums[c]  # cDist0
            needed = k - freeOne
            # Count how many 1's within distance<=1
            cDist1 = count_ones_in_range(c-1, c+1)
            # Count how many 1's within distance<=2
            cDist2Full = count_ones_in_range(c-2, c+2)
            cDist1Excl = cDist1 - freeOne  # distance=1 ones, excluding the free pickup if cDist0=1
            if needed <= 0:
                # Already have enough because we picked up the freeOne and k=1, or k was 1
                cost = 0
            else:
                used1 = min(cDist1Excl, needed)
                leftover = needed - used1
                cDist2 = cDist2Full - cDist1  # those exactly at distance=2
                # Check feasibility
                if leftover > cDist2 + maxChanges:
                    cost = INF  # can't pick enough ones from distance=2 + new ones
                else:
                    # cost = used1*(1 move each) + leftover*(2 moves each) = used1 + 2*leftover
                    cost = used1 + 2*leftover
            best = min(best, cost)

        return best