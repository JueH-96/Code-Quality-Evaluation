from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        """
        For every i we may pick any integer in the interval
        [start[i] , start[i] + d].  
        Let T be the minimum distance we want to guarantee between every
        pair of chosen numbers.  
        We have to find the largest T for which such an assignment is
        possible.

        Observation
        -----------
        After sorting the intervals by their left border, if an
        assignment with distance ≥ T exists, the greedy strategy
        “always pick the left-most legal value” succeeds as well:
        choosing the earliest possible point can only give *more*
        freedom to the remaining yet-to-be-chosen intervals.

        Feasibility check for a given T
        --------------------------------
        prev = previously chosen number
        for every interval [L , R]:
            we must place the next number at
                    x = max(L , prev + T)
            it is feasible for this interval iff x ≤ R.
            set prev = x and continue.

        The whole check is O(n).

        Binary search
        -------------
        The answer is monotone with respect to feasibility, hence we
        binary-search the largest T.
            lower bound : 0
            upper bound : (max(start)+d) - min(start)   (+1 for
                         exclusive upper end)
        Overall complexity:  O(n log range) ,   range ≤ 2·10⁹,
        which is easily fast enough for n ≤ 1e5.
        """
        n = len(start)
        starts = sorted(start)
        ends   = [s + d for s in starts]

        # test whether distance `dist` is achievable
        def feasible(dist: int) -> bool:
            prev = float('-inf')               # nothing chosen yet
            for L, R in zip(starts, ends):
                x = max(L, prev + dist)        # earliest legal value
                if x > R:                      # cannot place inside interval
                    return False
                prev = x
            return True

        # search the biggest distance that is feasible
        lo = 0
        hi = (max(start) + d - min(start)) + 1   # exclusive upper bound
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                lo = mid + 1                     # try to enlarge distance
            else:
                hi = mid                         # need to lower distance
        return lo - 1                            # last feasible value