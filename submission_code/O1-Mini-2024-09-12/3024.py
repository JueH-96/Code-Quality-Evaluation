class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Build the failure function for KMP
        def build_kmp(pattern):
            m = len(pattern)
            lps = [0] * m
            length = 0
            for i in range(1, m):
                while length > 0 and pattern[i] != pattern[length]:
                    length = lps[length -1]
                if pattern[i] == pattern[length]:
                    length +=1
                    lps[i] = length
            return lps
        
        # KMP search to find all start positions of pattern in text
        def kmp_search(text, pattern, lps):
            positions = []
            n_text = len(text)
            m = len(pattern)
            i = j =0
            while i < n_text:
                while j >0 and text[i] != pattern[j]:
                    j = lps[j-1]
                if text[i] == pattern[j]:
                    j +=1
                    if j == m:
                        start = i - m +1
                        if start < n:
                            positions.append(start)
                        j = lps[j-1]
                i +=1
            return positions
        
        lps = build_kmp(t)
        concatenated = s + s
        positions = kmp_search(concatenated, t, lps)
        
        M = []
        for p in positions:
            m = (n - p) %n
            M.append(m)
        
        if not M:
            return 0
        
        c0 = M.count(0)
        c1 = len(M) - c0
        
        pow_n_1 = pow(n-1, k, MOD)
        pow_neg1_k = 1 if k %2 ==0 else MOD-1
        pow_neg1_kp1 = MOD - pow_neg1_k
        inv_n = pow(n, MOD-2, MOD)
        
        term1 = (c0 * (pow_n_1 + pow_neg1_k * (n-1))) % MOD
        term2 = (c1 * (pow_n_1 + pow_neg1_kp1)) % MOD
        answer = (term1 + term2) * inv_n % MOD
        return answer