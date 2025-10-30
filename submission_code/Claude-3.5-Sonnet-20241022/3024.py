class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Find number of rotations that convert s to t
        def kmp(pattern, text):
            m = len(pattern)
            n = len(text)
            
            # Compute LPS array
            lps = [0] * m
            i = 1
            length = 0
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
            
            # Find matches
            count = 0
            i = j = 0
            while i < n:
                if pattern[j] == text[i]:
                    i += 1
                    j += 1
                if j == m:
                    count += 1
                    j = lps[j-1]
                elif i < n and pattern[j] != text[i]:
                    if j != 0:
                        j = lps[j-1]
                    else:
                        i += 1
            return count
        
        # Count how many rotations convert s to t
        count = kmp(t, s + s[:-1])
        
        # Matrix exponentiation for calculating result
        def matrix_multiply(a, b):
            return [[(a[0][0]*b[0][0] + a[0][1]*b[1][0]) % MOD, 
                    (a[0][0]*b[0][1] + a[0][1]*b[1][1]) % MOD],
                   [(a[1][0]*b[0][0] + a[1][1]*b[1][0]) % MOD, 
                    (a[1][0]*b[0][1] + a[1][1]*b[1][1]) % MOD]]
        
        def matrix_power(matrix, power):
            if power == 0:
                return [[1, 0], [0, 1]]
            if power == 1:
                return matrix
            
            half = matrix_power(matrix, power // 2)
            if power % 2 == 0:
                return matrix_multiply(half, half)
            return matrix_multiply(matrix_multiply(half, half), matrix)
        
        # Base matrix for transition
        base_matrix = [[n-1, 1], [n-1, 1]]
        result_matrix = matrix_power(base_matrix, k)
        
        if s == t:
            return result_matrix[0][0] % MOD
        else:
            return result_matrix[0][1] % MOD