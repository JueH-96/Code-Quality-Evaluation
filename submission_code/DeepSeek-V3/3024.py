MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        # Precompute the shifts where s shifted by l equals t
        shifts = []
        for l in range(1, n):
            if s[-l:] + s[:-l] == t:
                shifts.append(l)
        if s == t:
            shifts.append(0)
        
        if not shifts:
            return 0
        
        # Precompute the number of ways to reach each shift in k steps
        # We can model this as a graph where each node is a shift, and edges represent possible operations
        # The problem reduces to finding the number of paths of length k from the initial state to the target state
        
        # Since the number of shifts is small (up to n), we can use matrix exponentiation to compute the number of ways
        # Let's represent the transitions as a matrix
        
        # Initialize the transition matrix
        # For each shift l, the possible next shifts are l + x, where x is a shift that can be applied
        # But since the operation is cyclic, the shifts are modulo n
        
        # Instead of building the full transition matrix, we can note that the number of ways to reach a shift l in k steps is the same as the number of ways to reach l in k steps, considering the cyclic nature
        
        # We can use dynamic programming with memoization to compute the number of ways
        
        # Initialize dp[k][l] as the number of ways to reach shift l in k steps
        # dp[0][0] = 1 (starting with no shift)
        # For each step, dp[i][l] = sum(dp[i-1][m] for all m where m + x = l mod n)
        
        # However, with k up to 1e15, we need a more efficient approach
        
        # Instead, we can use the fact that the transitions form a linear recurrence, and we can use matrix exponentiation to compute the number of ways in O(log k) time
        
        # Let's represent the shifts as a list of possible l values
        # We need to find the number of ways to reach any of these l values in k steps
        
        # The transition matrix can be represented as follows:
        # For each shift l, the next shift can be any l' such that l' = (l + x) mod n, where x is a shift that can be applied
        
        # Since the number of shifts is small, we can precompute the transition matrix and use matrix exponentiation
        
        # First, let's find all possible shifts that can be applied
        possible_shifts = set()
        for l in range(1, n):
            possible_shifts.add(l)
        
        # Now, for each shift l, we can compute the next possible shifts
        # For each l, the next shift is (l + x) mod n, where x is in possible_shifts
        
        # We can represent the transitions as a graph and use matrix exponentiation to compute the number of ways to reach the target shifts in k steps
        
        # Let's create a mapping from shifts to indices
        shift_to_index = {l: i for i, l in enumerate(shifts)}
        index_to_shift = {i: l for i, l in enumerate(shifts)}
        m = len(shifts)
        
        # Initialize the transition matrix
        transition = [[0] * m for _ in range(m)]
        for i in range(m):
            l = index_to_shift[i]
            for x in possible_shifts:
                new_l = (l + x) % n
                if new_l in shift_to_index:
                    j = shift_to_index[new_l]
                    transition[i][j] += 1
        
        # Initialize the initial state vector
        # Initially, the shift is 0 (no shift)
        initial = [0] * m
        if 0 in shift_to_index:
            initial[shift_to_index[0]] = 1
        
        # Now, we need to raise the transition matrix to the power of k and multiply it by the initial vector
        # To do this efficiently, we use matrix exponentiation
        
        def matrix_mult(a, b):
            res = [[0] * m for _ in range(m)]
            for i in range(m):
                for j in range(m):
                    for l in range(m):
                        res[i][j] = (res[i][j] + a[i][l] * b[l][j]) % MOD
            return res
        
        def matrix_pow(mat, power):
            result = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
            while power > 0:
                if power % 2 == 1:
                    result = matrix_mult(result, mat)
                mat = matrix_mult(mat, mat)
                power = power // 2
            return result
        
        transition_k = matrix_pow(transition, k)
        
        # Multiply the transition_k by the initial vector
        final = [0] * m
        for i in range(m):
            for j in range(m):
                final[j] = (final[j] + initial[i] * transition_k[i][j]) % MOD
        
        # Now, sum the final values for all shifts that are in the target shifts
        total = 0
        for l in shifts:
            if l in shift_to_index:
                total = (total + final[shift_to_index[l]]) % MOD
        
        return total