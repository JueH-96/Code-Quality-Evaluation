class Solution:
    def waysToReachStair(self, k: int) -> int:
        # We start on stair 1; net displacement needed is (k - 1).
        # Suppose we make m "up" moves of sizes 2^0,2^1,...,2^(m-1),
        # so total up = S = 2^m - 1. Let D be number of down moves (each -1).
        # We need S - D = k - 1  =>  D = S - (k-1).
        # The downs cannot be consecutive, so in the sequence of m ups and D downs
        # the downs must fit into the m+1 "gaps" (before/between/after ups).
        # Number of ways for given (m,D) is C(m+1, D), provided 0 <= D <= m+1.
        #
        # We sum over all m>=0 for which D is in [0..m+1].
        #
        # Since 2^m grows fast, D will exceed m+1 once m is beyond ~log2(k)+2.
        
        import math
        
        target = k - 1
        res = 0
        # loop m from 0 up to, say, 32 (enough for k up to 1e9)
        for m in range(0, 35):
            S = (1 << m) - 1
            D = S - target
            if D < 0:
                # not enough up-sum yet, need more m
                continue
            if D > m + 1:
                # too many downs to fit without adjacency
                # and for larger m, D only grows larger, so we can break
                break
            # count ways to choose D gaps among m+1
            res += math.comb(m + 1, D)
        return res