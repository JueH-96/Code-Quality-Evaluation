MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        text = s + s
        pattern = t
        occ = self.kmp_search(text, pattern)
        m = sum(1 for idx in occ if 0 <= idx < n)
        
        M = [[0, n - 1], [1, n - 2]]
        Mk = self.mat_pow_2x2(M, k, MOD)
        A_k = Mk[0][0]
        B_k = Mk[1][0]
        
        if s == t:
            non_zero_count = m - 1
            ans = (A_k + non_zero_count * B_k) % MOD
        else:
            non_zero_count = m
            ans = (non_zero_count * B_k) % MOD
        return ans

    def compute_lps(self, pattern):
        n = len(pattern)
        lps = [0] * n
        length = 0
        i = 1
        while i < n:
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

    def kmp_search(self, text, pattern):
        n_pat = len(pattern)
        if n_pat == 0:
            return []
        lps = self.compute_lps(pattern)
        i = 0
        j = 0
        occurrences = []
        while i < len(text):
            if j < n_pat and text[i] == pattern[j]:
                i += 1
                j += 1
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
            if j == n_pat:
                occurrences.append(i - j)
                j = lps[j - 1]
        return occurrences

    def mat_mult_2x2(self, A, B, mod):
        a00 = (A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod
        a01 = (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod
        a10 = (A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod
        a11 = (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod
        return [[a00, a01], [a10, a11]]

    def mat_pow_2x2(self, matrix, power, mod):
        result = [[1, 0], [0, 1]]
        base = matrix
        while power:
            if power & 1:
                result = self.mat_mult_2x2(result, base, mod)
            base = self.mat_mult_2x2(base, base, mod)
            power >>= 1
        return result