class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        # dp_prev[c][p]: min flips up to previous pos, ending with char c (0 or 1)
        # and run-length parity p (0 = even length so far, 1 = odd length so far)
        INF = 10**18
        dp_prev = [[INF, INF], [INF, INF]]
        
        # Initialize at position 0
        bit0 = int(s[0])
        for c in (0, 1):
            flips = 0 if c == bit0 else 1
            # first run has length 1 => parity = 1 (odd)
            dp_prev[c][1] = flips
        
        # Iterate positions 1..n-1
        for i in range(1, n):
            bit = int(s[i])
            dp_curr = [[INF, INF], [INF, INF]]
            for prev_c in (0, 1):
                for prev_p in (0, 1):
                    cost_so_far = dp_prev[prev_c][prev_p]
                    if cost_so_far >= INF:
                        continue
                    # Try setting s[i] to c
                    for c in (0, 1):
                        flips = cost_so_far + (0 if c == bit else 1)
                        if c == prev_c:
                            # continue the run: flip parity
                            new_p = prev_p ^ 1
                            if flips < dp_curr[c][new_p]:
                                dp_curr[c][new_p] = flips
                        else:
                            # start new run: only allowed if previous run was even-length
                            if prev_p == 0:
                                new_p = 1  # new run length = 1
                                if flips < dp_curr[c][new_p]:
                                    dp_curr[c][new_p] = flips
                            # if prev_p==1 => can't break an odd run
            dp_prev = dp_curr
        
        # We must end with runs of even length => parity 0 for last run
        ans = min(dp_prev[0][0], dp_prev[1][0])
        return ans