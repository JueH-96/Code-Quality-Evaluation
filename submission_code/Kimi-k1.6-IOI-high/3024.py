class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if len(t) != n:
            return 0
        
        combined = s + s
        if t not in combined:
            return 0
        
        def compute_lps(pattern):
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps
        
        def kmp_search(text, pattern):
            m = len(pattern)
            if m == 0:
                return set()
            lps = compute_lps(pattern)
            i = j = 0
            valid_starts = set()
            text_len = len(text)
            while i < text_len:
                if text[i] == pattern[j]:
                    i += 1
                    j += 1
                    if j == m:
                        start = i - j
                        if start < n:
                            valid_starts.add(start)
                        j = lps[j - 1]
                else:
                    if j != 0:
                        j = lps[j - 1]
                    else:
                        i += 1
            return valid_starts
        
        valid_rs = kmp_search(combined, t)
        m = len(valid_rs)
        c = 1 if 0 in valid_rs else 0
        
        pow_n_1 = pow(n - 1, k, MOD)
        pow_neg1 = 1 if k % 2 == 0 else MOD - 1
        inv_n = pow(n, MOD - 2, MOD)
        
        a_k = (pow_n_1 + (n - 1) * pow_neg1) * inv_n % MOD
        b_k = (pow_n_1 - pow_neg1) * inv_n % MOD
        
        ans = (c * a_k + (m - c) * b_k) % MOD
        return ans