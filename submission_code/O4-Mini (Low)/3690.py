class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        if numOps >= n:
            return 1  # can break every character
        
        # helper: can we achieve max-run <= L with at most numOps flips?
        def can(L: int) -> bool:
            INF = n + 1
            # dp[run_len][bit] = min flips to reach here at current position
            # run_len from 1..L, bit in {0,1}
            dp = [[INF] * 2 for _ in range(L+1)]
            
            # initialize at position 0
            orig = int(s[0])
            # no flip
            dp[1][orig] = 0
            # flip
            dp[1][orig^1] = 1
            
            # iterate positions 1..n-1
            for i in range(1, n):
                orig = int(s[i])
                dp2 = [[INF] * 2 for _ in range(L+1)]
                for run_len in range(1, L+1):
                    for b in (0,1):
                        prev_cost = dp[run_len][b]
                        if prev_cost > numOps:
                            continue
                        # try no flip
                        b2 = orig
                        cost2 = prev_cost
                        if b2 == b:
                            nl = run_len + 1
                            if nl <= L:
                                if cost2 < dp2[nl][b2]:
                                    dp2[nl][b2] = cost2
                        else:
                            # new run
                            if cost2 < dp2[1][b2]:
                                dp2[1][b2] = cost2
                        # try flip
                        b2 = orig ^ 1
                        cost2 = prev_cost + 1
                        if cost2 <= numOps:
                            if b2 == b:
                                nl = run_len + 1
                                if nl <= L and cost2 < dp2[nl][b2]:
                                    dp2[nl][b2] = cost2
                            else:
                                if cost2 < dp2[1][b2]:
                                    dp2[1][b2] = cost2
                dp = dp2
            
            # after last pos, check any state <= numOps
            for run_len in range(1, L+1):
                if dp[run_len][0] <= numOps or dp[run_len][1] <= numOps:
                    return True
            return False
        
        # binary search on answer L
        lo, hi = 1, n
        ans = n
        while lo <= hi:
            mid = (lo + hi) // 2
            if can(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans