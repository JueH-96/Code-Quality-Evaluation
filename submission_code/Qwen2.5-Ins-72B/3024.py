class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Function to compute the number of rotations needed to transform s to t
        def rotations_needed(s, t):
            concatenated = s + s
            index = concatenated.find(t)
            if index == -1:
                return -1
            return index
        
        # Compute the number of rotations needed
        r = rotations_needed(s, t)
        if r == -1:
            return 0
        
        # Compute the number of valid rotations
        valid_rotations = (k % n) == r
        if k % 2 == 0:
            return (n - 1) * valid_rotations % MOD
        else:
            return valid_rotations % MOD
        
        # Compute the number of ways to achieve the required rotations in k steps
        def matrix_exponentiation(k, n):
            F = [[0, 1], [1, n-1]]
            result = [[1, 0], [0, 1]]
            while k > 0:
                if k % 2 == 1:
                    result = [[(result[0][0] * F[0][0] + result[0][1] * F[1][0]) % MOD,
                               (result[0][0] * F[0][1] + result[0][1] * F[1][1]) % MOD],
                              [(result[1][0] * F[0][0] + result[1][1] * F[1][0]) % MOD,
                               (result[1][0] * F[0][1] + result[1][1] * F[1][1]) % MOD]]
                F = [[(F[0][0] * F[0][0] + F[0][1] * F[1][0]) % MOD,
                      (F[0][0] * F[0][1] + F[0][1] * F[1][1]) % MOD],
                     [(F[1][0] * F[0][0] + F[1][1] * F[1][0]) % MOD,
                      (F[1][0] * F[0][1] + F[1][1] * F[1][1]) % MOD]]
                k //= 2
            return result
        
        # Compute the final result
        result_matrix = matrix_exponentiation(k, n)
        if r == 0:
            return result_matrix[0][0]
        else:
            return result_matrix[0][1]