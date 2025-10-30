MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        mod = MOD
        n = len(s)
        text = s + s
        occ = self.kmp_search(text, t)
        M = 0
        for idx in occ:
            if 0 <= idx < n:
                M += 1
        
        if M == 0:
            return 0
        
        I0 = 1 if s == t else 0
        base_val = I0 * n - M
        
        term1 = M * pow(n - 1, k, mod) % mod
        
        term2 = base_val % mod
        if k % 2 == 1:
            term2 = (-term2) % mod
        
        numerator = (term1 + term2) % mod
        inv_n = pow(n, mod - 2, mod)
        ans = numerator * inv_n % mod
        return ans

    def kmp_search(self, text, pattern):
        n_text = len(text)
        m = len(pattern)
        if m == 0:
            return []
        pi = [0] * m
        k_val = 0
        for q in range(1, m):
            while k_val > 0 and pattern[k_val] != pattern[q]:
                k_val = pi[k_val - 1]
            if pattern[k_val] == pattern[q]:
                k_val += 1
            else:
                k_val = 0
            pi[q] = k_val
        
        res = []
        q = 0
        for i in range(n_text):
            while q > 0 and pattern[q] != text[i]:
                q = pi[q - 1]
            if pattern[q] == text[i]:
                q += 1
            if q == m:
                start_index = i - m + 1
                res.append(start_index)
                q = pi[q - 1]
        return res