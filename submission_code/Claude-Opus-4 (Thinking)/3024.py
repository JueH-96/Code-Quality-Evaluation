class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Find all positions where t appears in s + s using KMP
        def kmp_search(text, pattern):
            def compute_lps(pattern):
                m = len(pattern)
                lps = [0] * m
                j = 0
                for i in range(1, m):
                    while j > 0 and pattern[i] != pattern[j]:
                        j = lps[j - 1]
                    if pattern[i] == pattern[j]:
                        j += 1
                    lps[i] = j
                return lps
            
            lps = compute_lps(pattern)
            positions = []
            j = 0
            for i in range(len(text)):
                while j > 0 and text[i] != pattern[j]:
                    j = lps[j - 1]
                if text[i] == pattern[j]:
                    j += 1
                if j == len(pattern):
                    positions.append(i - j + 1)
                    j = lps[j - 1]
            return positions
        
        # Find all occurrences of t in s + s
        s2 = s + s
        positions = kmp_search(s2, t)
        
        # Filter positions to be within [0, n)
        rotations = [pos for pos in positions if pos < n]
        
        if not rotations:
            return 0
        
        # Matrix exponentiation to compute number of ways
        def matrix_mult(A, B):
            return [
                [(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % MOD,
                 (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % MOD],
                [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % MOD,
                 (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % MOD]
            ]
        
        def matrix_pow(M, p):
            result = [[1, 0], [0, 1]]  # Identity matrix
            base = [row[:] for row in M]  # Copy M
            while p > 0:
                if p % 2 == 1:
                    result = matrix_mult(result, base)
                base = matrix_mult(base, base)
                p //= 2
            return result
        
        # Transition matrix: [[0, n-1], [1, n-2]]
        M = [[0, n - 1], [1, n - 2]]
        Mk = matrix_pow(M, k)
        
        # Extract a[k] and b[k] from M^k * [1, 0]^T
        a_k = Mk[0][0]
        b_k = Mk[1][0]
        
        result = 0
        for r in rotations:
            if r == 0:
                result = (result + a_k) % MOD
            else:
                result = (result + b_k) % MOD
        
        return result