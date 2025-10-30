class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        MOD = 10**9 + 7
        
        # Check if t is a rotation of s
        if t not in s + s:
            return 0
        
        # Find all positions where t appears as a rotation of s
        positions = []
        double_s = s + s
        for i in range(n):
            if double_s[i:i+n] == t:
                positions.append(i)
        
        # Count the number of ways
        # Let's think about this as a graph problem
        # We have n possible rotations (positions 0 to n-1)
        # From any position, we can go to n-1 other positions in one move
        
        # dp[i][j] = number of ways to reach position j in exactly i moves
        # But k is too large for direct DP
        
        # Let's use the fact that this forms a regular graph
        # From any position, we can reach any of the other n-1 positions
        
        # Let f(k, same) = number of ways to end at the same position after k moves
        # Let g(k, diff) = number of ways to end at a different specific position after k moves
        
        # Recurrence:
        # f(k+1, same) = (n-1) * g(k, diff)
        # g(k+1, diff) = f(k, same) + (n-2) * g(k, diff)
        
        # Base case:
        # f(0, same) = 1 (we start at the same position)
        # g(0, diff) = 0 (we can't be at a different position without moving)
        
        # We can solve this using matrix exponentiation
        # [f(k+1)]   [0     n-1] [f(k)]
        # [g(k+1)] = [1     n-2] [g(k)]
        
        def matrix_mult(A, B):
            return [
                [(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % MOD,
                 (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % MOD],
                [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % MOD,
                 (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % MOD]
            ]
        
        def matrix_pow(M, p):
            if p == 0:
                return [[1, 0], [0, 1]]
            if p == 1:
                return M
            
            if p % 2 == 0:
                half = matrix_pow(M, p // 2)
                return matrix_mult(half, half)
            else:
                return matrix_mult(M, matrix_pow(M, p - 1))
        
        # Transition matrix
        M = [[0, n-1], [1, n-2]]
        
        # Compute M^k
        result_matrix = matrix_pow(M, k)
        
        # Initial state: [1, 0] (at same position, not at different position)
        f_k = result_matrix[0][0] * 1 + result_matrix[0][1] * 0
        g_k = result_matrix[1][0] * 1 + result_matrix[1][1] * 0
        
        # Count the answer
        answer = 0
        for pos in positions:
            if pos == 0:
                # t is the same as s (no rotation needed)
                answer = (answer + f_k) % MOD
            else:
                # t is a different rotation of s
                answer = (answer + g_k) % MOD
        
        return answer