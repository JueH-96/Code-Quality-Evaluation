class Solution:
    def waysToReachStair(self, k: int) -> int:
        """
        We want the number of distinct sequences of operations that start on stair 1
        and eventually end on stair k, possibly passing through k multiple times in-between.
        
        Key observations and derivation:

        1) Each "up" operation increases the current stair i by 2^(jump), then jump += 1.
           If we perform n such up-operations in total (with jumps 0, 1, 2, ..., n-1),
           the total upward move is 2^0 + 2^1 + ... + 2^(n-1) = 2^n - 1 (for n>0).
           If n=0, no "up" operations occur, so the total upward move is 0.

        2) Each "down" operation decreases the current stair i by 1, cannot be used consecutively,
           and cannot be used on stair 0.

        3) We start at stair 1. If we do n ups in total, the net upward displacement is (2^n - 1) if n>0
           (or 0 if n=0). Let d be the total number of downs. Then our final position is:
               final_stair = 1 + (2^n - 1) - d   (if n>0)
           or
               final_stair = 1 - d               (if n=0)
           We want final_stair = k.

           For n > 0,  1 + (2^n - 1) - d = k  =>  2^n - d = k  =>  d = 2^n - k.
           For n = 0,  1 - d = k  =>  d = 1 - k; that only works for k = 0 or k = 1.

        4) Constraint: no two downs in a row, and never down on stair 0. 
           However, starting at stair 1 ensures that if we ever go down to stair 0,
           the next operation cannot be another down (it would be consecutive downs and also invalid at 0).
           This "no consecutive downs" condition ensures we cannot down from 0 again.

           In a sequence of n ups and d downs (total n+d moves), 
           "no two consecutive downs" is a classic combinatorial condition:
           the number of ways to interleave d downs among n ups with no two downs adjacent
           is C(n+1, d) provided d <= n+1.  (Think of placing d "D"s in the n+1 "gaps" around n "U"s.)

        5) Combining these:
           - For each n >= 1 where 2^n >= k:
               d = 2^n - k
               if d <= n+1, then we can choose C(n+1, d) ways to interleave those downs.
           - For n = 0, we get a valid sequence only if k = 0 or k = 1.

        6) We sum over all n that satisfy the above.  Since k <= 10^9, we only need n up to around 60
           (2^30 already exceeds 10^9, and we check further for the condition 2^n <= k + n + 1).

        This yields a finite sum of binomial coefficients, which we can safely compute in Python.
        """

        import math

        ways = 0
        # Check n from 0 through ~60 (safely above 2^30 for k up to 10^9)
        for n in range(61):
            power_of_two = 1 << n  # 2^n

            # If n == 0, then final position = 1 - d => we want 1 - d = k => d = 1-k
            # That is valid only if d >= 0 and no consecutive-down constraints are broken.
            # For k=0 => d=1 => "down once" is valid; for k=1 => d=0 => no moves is valid.
            if n == 0:
                if k == 0:
                    ways += 1  # one sequence: down once (1->0)
                elif k == 1:
                    ways += 1  # one sequence: do nothing (already at stair 1)
                continue
            
            # For n >= 1 => d = 2^n - k must be >= 0 to reach k (otherwise we overshoot negative)
            if power_of_two < k:
                continue
            d = power_of_two - k
            # We can have at most one down in each gap between ups (and possibly before/after),
            # so d must be <= n+1
            if d <= n + 1:
                ways += math.comb(n + 1, d)

        return ways