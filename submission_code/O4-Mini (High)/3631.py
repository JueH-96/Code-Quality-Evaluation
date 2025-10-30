class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        mod = 10**9 + 7
        L = len(s)
        # Precompute g(c): number of steps to reduce c to 1 by popcount operations.
        # We store dp_g[c] = g(c). For c=0, set large so it's never allowed.
        dp_g = [0] * (L + 1)
        dp_g[0] = k + 1  # make sure 0 is never considered reducible
        dp_g[1] = 0
        # Python 3.8+: use bit_count, else fallback to bin count
        for c in range(2, L + 1):
            pc = c.bit_count()
            dp_g[c] = 1 + dp_g[pc]
        # Collect allowed popcounts c such that any x with popcount c has
        # g(x)=1+g(c) <= k, i.e. g(c) <= k-1.  c=1 gives g(x)=0 or 1<=k => included.
        limit = k - 1
        allowed = [False] * (L + 1)
        for c in range(1, L + 1):
            if dp_g[c] <= limit:
                allowed[c] = True

        # DP over bits of s to count numbers < n having popcount in allowed.
        # eq[c] = # ways to build prefix equal to s so far with c ones
        # lt[c] = # ways to build prefix strictly less than s so far with c ones
        eq = [0] * (L + 1)
        lt = [0] * (L + 1)
        eq[0] = 1

        for pos in range(L):
            s_bit = int(s[pos])
            new_eq = [0] * (L + 1)
            new_lt = [0] * (L + 1)
            # transitions from states already less than s
            for c in range(0, L + 1):
                v = lt[c]
                if not v:
                    continue
                # once less, can put 0 or 1 freely
                # put 0
                new_lt[c] = (new_lt[c] + v) % mod
                # put 1
                new_lt[c + 1] = (new_lt[c + 1] + v) % mod
            # transitions from states equal to s so far
            if s_bit == 0:
                # must put b=0 to stay <= s[pos]
                for c in range(0, L + 1):
                    v = eq[c]
                    if not v:
                        continue
                    # put 0 -> stay equal
                    new_eq[c] = (new_eq[c] + v) % mod
            else:
                # s_bit == 1: we can put b=0 -> become less, or b=1 -> stay equal
                for c in range(0, L + 1):
                    v = eq[c]
                    if not v:
                        continue
                    # put 0 -> become less
                    new_lt[c] = (new_lt[c] + v) % mod
                    # put 1 -> stay equal
                    new_eq[c + 1] = (new_eq[c + 1] + v) % mod
            eq, lt = new_eq, new_lt

        # Now lt[c] holds # of integers in [0, n) with popcount = c.
        # Sum over allowed c; note c=0 is not allowed, so we don't count 0.
        ans = 0
        for c in range(1, L + 1):
            if allowed[c]:
                ans = (ans + lt[c]) % mod

        return ans