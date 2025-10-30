class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        cyclic_permutations = set()
        current_s = s
        for _ in range(n):
            cyclic_permutations.add(current_s)
            current_s = current_s[1:] + current_s[0]
        
        permutation_list = sorted(list(cyclic_permutations))
        n_perms = len(permutation_list)
        s_to_index = {permutation_list[i]: i for i in range(n_perms)}
        
        if t not in s_to_index:
            return 0
        
        target_index = s_to_index[t]
        start_index = s_to_index[s]
        
        transition_matrix = [[0] * n_perms for _ in range(n_perms)]
        
        for i in range(n_perms):
            current_string = permutation_list[i]
            for l in range(1, n):
                suffix = current_string[n-l:]
                prefix = current_string[:n-l]
                next_string = suffix + prefix
                next_index = s_to_index[next_string]
                transition_matrix[i][next_index] += 1
                
        def multiply_matrices(A, B, mod):
            C = [[0] * n_perms for _ in range(n_perms)]
            for i in range(n_perms):
                for j in range(n_perms):
                    for r in range(n_perms):
                        C[i][j] = (C[i][j] + A[i][r] * B[r][j]) % mod
            return C
            
        def power_matrix(matrix, power, mod):
            identity_matrix = [[0] * n_perms for _ in range(n_perms)]
            for i in range(n_perms):
                identity_matrix[i][i] = 1
            
            result_matrix = identity_matrix
            current_power_matrix = matrix
            
            while power > 0:
                if power % 2 == 1:
                    result_matrix = multiply_matrices(result_matrix, current_power_matrix, mod)
                current_power_matrix = multiply_matrices(current_power_matrix, current_power_matrix, mod)
                power //= 2
                
            return result_matrix
            
        if k == 0:
            return 1 if s == t else 0
            
        if k <= 0:
            return 0

        result_power_matrix = power_matrix(transition_matrix, k, 10**9 + 7)
        return result_power_matrix[start_index][target_index]