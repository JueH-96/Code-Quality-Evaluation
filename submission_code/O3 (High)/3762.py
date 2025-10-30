from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        
        # ------------------------------------------------------------------ #
        #  Feasibility test : can we make the minimum score at least target? #
        # ------------------------------------------------------------------ #
        def possible(target: int) -> bool:
            # number of visits that every index still needs
            need = [(target + p - 1) // p for p in points]   # ceil(target / points[i])
            
            steps = 1               # first mandatory move: -1 -> 0
            need[0] -= 1            # we already visited index 0 once
            if need[0] < 0:
                need[0] = 0         # we never need negative visits
            
            # right-most index whose requirement is still > 0
            last = -1
            for i in range(n - 1, -1, -1):
                if need[i] > 0:
                    last = i
                    break
            
            # all requirements already satisfied
            if last == -1:
                return steps <= m
            
            # -------------------------------------------------------------- #
            #  Process indices 0 … last-1, staying at i while it needs visits #
            # -------------------------------------------------------------- #
            for i in range(last):
                b = need[i]                     # visits that index i still needs
                if b:                           # do the required bounces
                    steps += 2 * b              # each extra visit: out and back (2 moves)
                    need[i] = 0
                    need[i + 1] -= b            # every bounce visits i+1 once
                    if need[i + 1] < 0:
                        need[i + 1] = 0
                # after satisfying i we are still at i
                if i + 1 < last:                # we must proceed to the right
                    steps += 1                  # single step i -> i+1
                    need[i + 1] -= 1            # that step also visits i+1
                    if need[i + 1] < 0:
                        need[i + 1] = 0
                if steps > m:                   # early exit
                    return False
            
            # -------------------------------------------------------------- #
            #  Handle the very last necessary index `last`                   #
            # -------------------------------------------------------------- #
            remain = need[last]                 # visits still required for `last`
            if remain < 0:
                remain = 0
            
            if last == 0:                       # we are already at index 0
                steps += 2 * remain             # each extra visit takes two moves (0 <-> 1)
            else:                               # currently at `last - 1`
                if remain:                      # need at least one visit to `last`
                    steps += 2 * remain - 1     # first time: one move, afterwards bounces
            
            return steps <= m
        
        # -------------------------------------------------------------- #
        #  Binary search for the largest achievable minimum score        #
        # -------------------------------------------------------------- #
        lo, hi = 0, min(points) * m             # upper bound is safe
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if possible(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans