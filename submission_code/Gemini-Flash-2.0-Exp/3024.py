class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        mod = 10**9 + 7

        def solve():
            if s == t:
                same = 1
            else:
                same = 0

            dp = {}

            def calculate(curr_s, moves):
                if moves == 0:
                    if curr_s == t:
                        return 1
                    else:
                        return 0
                
                if (curr_s, moves) in dp:
                    return dp[(curr_s, moves)]
                
                ways = 0
                for l in range(1, n):
                    new_s = curr_s[-l:] + curr_s[:-l]
                    ways = (ways + calculate(new_s, moves - 1)) % mod
                
                dp[(curr_s, moves)] = ways
                return ways

            return calculate(s, k)
        
        def matrix_exponentiation():
            if s == t:
                same = 1
            else:
                same = 0
            
            diff = n - 1 - same
            
            def matrix_multiply(A, B):
                C = [[0, 0], [0, 0]]
                for i in range(2):
                    for j in range(2):
                        for l in range(2):
                            C[i][j] = (C[i][j] + A[i][l] * B[l][j]) % mod
                return C
            
            def matrix_power(M, p):
                result = [[0, 0], [0, 0]]
                result[0][0] = 1
                result[1][1] = 1
                
                while p > 0:
                    if p % 2 == 1:
                        result = matrix_multiply(result, M)
                    M = matrix_multiply(M, M)
                    p //= 2
                return result
            
            M = [[diff, same], [same, diff]]
            M = matrix_power(M, k)
            
            if s == t:
                return M[0][0]
            else:
                return M[1][0]
        
        return matrix_exponentiation()