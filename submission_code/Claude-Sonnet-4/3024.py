class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Check if t is a rotation of s
        if t not in s + s:
            return 0
        
        # Count how many rotations of s equal t
        rotations_to_t = 0
        for i in range(n):
            if s[i:] + s[:i] == t:
                rotations_to_t += 1
        
        # If s == t initially
        if s == t:
            # We need to count ways to return to original position in k steps
            # This is a classic problem: ways to return to start in k steps on a cycle
            # Using the formula for random walk on cycle
            
            # dp[i] = number of ways to be at original position after i steps
            # We can use matrix exponentiation, but there's a direct formula
            
            # For a cycle of length n-1 (since we have n-1 possible moves),
            # the number of ways to return to start in k steps is:
            # ((n-2)^k + (-1)^k) / (n-1) if we consider all positions equally likely
            
            # But we need exact counting:
            # Let f(k) = ways to be at start after k steps
            # Let g(k) = ways to be at any non-start position after k steps
            # f(k) + (n-1)*g(k) = (n-1)^k (total ways)
            # f(k+1) = (n-2)*g(k) (from non-start positions)
            # g(k+1) = f(k) + (n-2)*g(k) (from start + from other non-start)
            
            # Solving: f(k) = ((n-2)^k + (-1)^k * (n-2)) / (n-1)
            # But this doesn't work for our exact problem
            
            # Let's use the correct recurrence:
            # ways[i] = ways to reach target in exactly i operations
            # If currently at target: can go to any of (n-1) positions
            # If currently not at target: can go to target in rotations_to_t ways, or stay non-target
            
            # Use matrix exponentiation approach
            return self.matrixPower(n, rotations_to_t, k, MOD)
        else:
            # s != t, so we start at non-target and want to reach target
            return self.matrixPower(n, rotations_to_t, k, MOD, start_at_target=False)
    
    def matrixPower(self, n, rotations_to_t, k, MOD, start_at_target=True):
        # State: [at_target, not_at_target]
        # Transition matrix:
        # From target: can go to target in 0 ways, non-target in (n-1) ways
        # From non-target: can go to target in rotations_to_t ways, non-target in (n-1-rotations_to_t) ways
        
        # Matrix: [[0, rotations_to_t], [n-1, n-1-rotations_to_t]]
        matrix = [[0, rotations_to_t], [n-1, (n-1-rotations_to_t) % MOD]]
        
        # Initial state
        if start_at_target:
            state = [1, 0]  # [at_target, not_at_target]
        else:
            state = [0, 1]
        
        # Matrix exponentiation
        result_matrix = self.matrixPowerHelper(matrix, k, MOD)
        
        # Apply to initial state
        result = 0
        for i in range(2):
            result = (result + state[i] * result_matrix[0][i]) % MOD
        
        return result
    
    def matrixPowerHelper(self, matrix, power, MOD):
        if power == 0:
            return [[1, 0], [0, 1]]  # Identity matrix
        
        if power == 1:
            return [[matrix[0][0] % MOD, matrix[0][1] % MOD], 
                   [matrix[1][0] % MOD, matrix[1][1] % MOD]]
        
        if power % 2 == 0:
            half = self.matrixPowerHelper(matrix, power // 2, MOD)
            return self.matrixMultiply(half, half, MOD)
        else:
            return self.matrixMultiply(matrix, self.matrixPowerHelper(matrix, power - 1, MOD), MOD)
    
    def matrixMultiply(self, A, B, MOD):
        return [[(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % MOD,
                 (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % MOD],
                [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % MOD,
                 (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % MOD]]