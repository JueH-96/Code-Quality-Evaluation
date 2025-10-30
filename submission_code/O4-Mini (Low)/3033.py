class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # collect mismatch positions
        p = [i for i in range(len(s1)) if s1[i] != s2[i]]
        m = len(p)
        if m & 1:
            return -1
        
        # Precompute A[k] = cost to pair p[k] with p[k+1] greedily
        # which is min(x, distance)
        A = [0] * (m - 1)
        for k in range(m - 1):
            dist = p[k+1] - p[k]
            A[k] = dist if dist < x else x
        
        # Build prefix sums for even and odd starting indices
        # ps[0][i] = sum of A at even indices < i with step 2
        # ps[1][i] = sum of A at odd indices < i with step 2
        ps = [[0] * (m) for _ in range(2)]
        for parity in (0, 1):
            s = 0
            for i in range(parity, m-1, 2):
                s += A[i]
                ps[parity][i] = s
            # fill forward so that ps[parity][j] holds the sum up to j or last <= j
            last = 0
            for j in range(m):
                if ps[parity][j] != 0:
                    last = ps[parity][j]
                ps[parity][j] = last
        
        # helper to get sum A[l] + A[l+2] + ... up to A[r-1]
        def seg_cost(l, r):
            # segment p[l..r], with r-l+1 even, use greedy adjacent pairing
            # we sum A[l] + A[l+2] + ... + A[r-1]
            if l > r:
                return 0
            parity = l & 1
            # we want sum from indices parity positions starting at l upto r-1
            # that's ps[parity][r-1] minus ps[parity][l-2]
            hi = ps[parity][r-1]
            lo = ps[parity][l-2] if l-2 >= 0 else 0
            return hi - lo
        
        # DP[i] = min cost to match first i mismatches (i is even)
        INF = 10**18
        dp = [INF] * (m+1)
        dp[0] = 0
        
        # fill dp
        for i in range(2, m+1, 2):
            # try closing last two by greedy adjacent
            dp[i] = dp[i-2] + A[i-2]
            # try pairing p[j] with p[i-1] for all j <= i-2, j even to keep segments even
            for j in range(0, i-2, 2):
                # cost to match p[j] with p[i-1] via best of type1 or chain
                pair_cost = min(x, p[i-1] - p[j])
                # cost to match internal mismatches between j+1 and i-2 greedily
                internal = seg_cost(j+1, i-2)
                dp[i] = min(dp[i], dp[j] + pair_cost + internal)
        
        return dp[m]