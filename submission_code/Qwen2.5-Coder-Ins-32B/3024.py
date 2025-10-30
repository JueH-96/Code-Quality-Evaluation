class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Check if t can be obtained by rotating s
        if t not in (s + s):
            return 0
        
        # Find all valid rotations that transform s into t
        valid_rotations = [i for i in range(1, n) if s[i:] + s[:i] == t]
        m = len(valid_rotations)
        
        # If no valid rotations, return 0
        if m == 0:
            return 0
        
        # If k is 0, check if s is already t
        if k == 0:
            return 1 if s == t else 0
        
        # Matrix exponentiation to find the number of ways
        def matrix_mult(A, B, mod):
            return [[(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod, (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod],
                    [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod, (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod]]
        
        def matrix_pow(matrix, power, mod):
            result = [[1, 0], [0, 1]]  # Identity matrix
            base = matrix
            while power:
                if power % 2 == 1:
                    result = matrix_mult(result, base, mod)
                base = matrix_mult(base, base, mod)
                power //= 2
            return result
        
        # Transition matrix for the problem
        transition_matrix = [[m - 1, m], [m, m - 1]]
        
        # Raise the transition matrix to the power of k
        result_matrix = matrix_pow(transition_matrix, k, MOD)
        
        # If s == t, we start in the state where we don't need to rotate
        if s == t:
            return result_matrix[0][0]
        else:
            return result_matrix[1][0]