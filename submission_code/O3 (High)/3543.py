class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        
        # Quick short-circuit: k larger than the whole string ⇒
        # every substring automatically fulfils the constraint.
        if k >= n:
            return n * (n + 1) // 2
        
        # Prefix sums of zeroes and ones
        pref0 = [0] * (n + 1)
        pref1 = [0] * (n + 1)
        for i, ch in enumerate(s, 1):          # 1-based indices for convenience
            pref0[i] = pref0[i - 1] + (ch == '0')
            pref1[i] = pref1[i - 1] + (ch == '1')
        
        res = 0
        # Enumerate every substring s[l-1 : r]  (1 ≤ l ≤ r ≤ n)
        for l in range(1, n + 1):
            for r in range(l, n + 1):
                zeros = pref0[r] - pref0[l - 1]
                ones  = pref1[r] - pref1[l - 1]
                if zeros <= k or ones <= k:
                    res += 1
        return res