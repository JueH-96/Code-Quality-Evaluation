from typing import List

MOD = 1_000_000_007

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # K = how many extra elements (besides the fixed min / max one)
        K = k - 1                       # 0 ≤ K ≤ 69
        nums.sort()                     # work on the ordered array

        # ------------------------------------------------------------------
        # Pre–compute modular inverses of 1 … K  (needed for fast binomials)
        # ------------------------------------------------------------------
        inv = [1] * (K + 1)
        for x in range(1, K + 1):
            inv[x] = pow(x, MOD - 2, MOD)

        # ------------------------------------------------------------------
        # S[m]  =  Σ_{t = 0 .. K}  C(m, t)   (0 ≤ m ≤ n)
        # Only the first K = 69 columns are required, so total work is O(n*K)
        # ------------------------------------------------------------------
        S = [0] * (n + 1)               # S[m] as defined above
        for m in range(n + 1):
            total = 1                   # C(m, 0)
            c = 1                       # current C(m, r)
            up_to = min(K, m)
            for r in range(1, up_to + 1):
                # C(m, r) = C(m, r-1) * (m - r + 1) / r  (mod MOD)
                c = c * (m - r + 1) % MOD
                c = c * inv[r] % MOD
                total += c
            S[m] = total % MOD

        # ------------------------------------------------------------------
        # Contribution of each element
        #   • as minimum:     choose any ≤K-1 elements from the right side
        #   • as maximum:     choose any ≤K-1 elements from the left side
        # ------------------------------------------------------------------
        ans = 0
        for i, val in enumerate(nums):
            cnt_min = S[n - i - 1]      # elements to the right of i
            cnt_max = S[i]              # elements to the left of i
            ans = (ans + val * (cnt_min + cnt_max)) % MOD

        return ans