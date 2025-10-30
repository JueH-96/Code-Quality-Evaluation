class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if n != len(t):
            return 0
        
        occurrences = self.kmp_search(t, s + s)
        R = [pos for pos in occurrences if pos < n]
        if not R:
            return 0
        
        count0 = R.count(0)
        count_rest = len(R) - count0
        
        inv_n = pow(n, MOD - 2, MOD)
        pow_n_minus_1_k = pow(n - 1, k, MOD)
        pow_neg1_k = 1 if k % 2 == 0 else MOD - 1
        
        # Calculate a_k
        term_a = (pow_n_minus_1_k - pow_neg1_k) % MOD
        a_k = (term_a * inv_n) % MOD
        
        # Calculate b_k
        term_b_part = ((n - 1) * pow_neg1_k) % MOD
        term_b = (pow_n_minus_1_k + term_b_part) % MOD
        b_k = (term_b * inv_n) % MOD
        
        ans = (count_rest * a_k + count0 * b_k) % MOD
        return ans
    
    def kmp_search(self, pattern, text):
        m = len(pattern)
        if m == 0:
            return []
        lps = [0] * m
        length = 0  # Length of the previous longest prefix suffix
        
        # Build the LPS array
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
        
        # Search for pattern in text
        occurrences = []
        n = len(text)
        i = 0  # Index for text
        j = 0  # Index for pattern
        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            if j == m:
                occurrences.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        return occurrences