class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        a = sorted(nums)
        
        # Collect all possible pairwise differences
        Dset = set()
        for i in range(n):
            for j in range(i+1, n):
                Dset.add(a[j] - a[i])
        D = sorted(Dset)
        
        # For each d in D, compute f(d) = # of k‑subsets with all pairwise gaps >= d
        f = []
        for d in D:
            # next_idx[i] = smallest j>i such that a[j]-a[i] >= d (or n if none)
            next_idx = [n] * n
            for i in range(n):
                j = i + 1
                while j < n and a[j] - a[i] < d:
                    j += 1
                next_idx[i] = j
            
            # dp2[i][c] = # of ways to pick c items from indices >= i
            #             such that any two picked differ by >= d
            dp2 = [[0] * (k+1) for _ in range(n+1)]
            dp2[n][0] = 1
            for i in range(n-1, -1, -1):
                dp2[i][0] = 1
                ni = next_idx[i]
                for c in range(1, k+1):
                    v = dp2[i+1][c] + dp2[ni][c-1]
                    # v < 2*mod, so single subtract suffices
                    if v >= mod:
                        v -= mod
                    dp2[i][c] = v
            f.append(dp2[0][k])
        
        # Now f[i] = # subsets with min‐gap >= D[i].
        # # subsets with min‐gap == D[i] = f[i] - f[i+1]
        res = 0
        m = len(D)
        for i, d in enumerate(D):
            c1 = f[i]
            c2 = f[i+1] if i+1 < m else 0
            cnt = c1 - c2
            if cnt < 0:
                cnt += mod
            res = (res + cnt * d) % mod
        
        return res