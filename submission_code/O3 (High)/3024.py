class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 1_000_000_007
        n = len(s)
        
        # ---------- 1. find all rotations of s that equal t ----------
        # every rotation of s is a substring of (s+s) that starts
        # in the first n positions.
        text = (s + s)[:-1]            # length = 2n-1  -> starts < n only
        pattern = t
        m = n
        
        # KMP prefix function
        def prefix_function(st: str):
            pi = [0] * len(st)
            for i in range(1, len(st)):
                j = pi[i-1]
                while j and st[i] != st[j]:
                    j = pi[j-1]
                if st[i] == st[j]:
                    j += 1
                pi[i] = j
            return pi
        
        combined = pattern + '#' + text      # separator not in alphabet
        pi = prefix_function(combined)
        
        total_good = 0   # number of shifts d with rotate(s,d)=t
        g0 = 0           # 1 if shift 0 is good
        for idx, val in enumerate(pi):
            if val == m:
                start_in_text = idx - 2*m      # see derivation in analysis
                if 0 <= start_in_text < n:     # only first n starts are real rotations
                    total_good += 1
                    if start_in_text == 0:     # shift 0
                        g0 = 1
        g_non0 = total_good - g0
        if total_good == 0:
            return 0
        
        # ---------- 2. compute number of sequences after k steps ----------
        inv_n = pow(n, MOD-2, MOD)                   # modular inverse of n  (MOD is prime)
        pow_n_minus_1 = pow(n-1, k, MOD)             # (n-1)^k   mod MOD
        sign = 1 if k % 2 == 0 else MOD-1            # (-1)^k  mod MOD (1 or -1)
        
        # ways to be at the same rotation (shift 0)
        same = (pow_n_minus_1 + (n-1) * sign) % MOD
        same = same * inv_n % MOD
        
        # ways to be at any specific different rotation (shift d != 0)
        diff = (pow_n_minus_1 - sign) % MOD
        diff = diff * inv_n % MOD
        
        # ---------- 3. answer ----------
        ans = (g0 * same + g_non0 * diff) % MOD
        return ans