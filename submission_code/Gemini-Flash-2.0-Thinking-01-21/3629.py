class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        counts = [0] * 26
        for char in s:
            counts[ord(char) - ord('a')] += 1
        
        transformation_matrix = [[0] * 26 for _ in range(26)]
        transformation_matrix[0][25] = 1
        transformation_matrix[1][0] = 1
        transformation_matrix[1][25] = 1
        for i in range(2, 26):
            transformation_matrix[i][i-1] = 1
            
        def multiply_matrices(A, B, mod):
            C = [[0] * 26 for _ in range(26)]
            for i in range(26):
                for j in range(26):
                    for k in range(26):
                        C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod
            return C
            
        def power(matrix, exp, mod):
            size = len(matrix)
            identity = [[0] * size for _ in range(size)]
            for i in range(size):
                identity[i][i] = 1
            result = identity
            while exp > 0:
                if exp % 2 == 1:
                    result = multiply_matrices(result, matrix, mod)
                matrix = multiply_matrices(matrix, matrix, mod)
                exp //= 2
            return result
            
        mod_val = 10**9 + 7
        if t == 0:
            return len(s)
            
        transformed_matrix = power(transformation_matrix, t, mod_val)
        
        next_counts = [0] * 26
        for i in range(26):
            for j in range(26):
                next_counts[i] = (next_counts[i] + transformed_matrix[i][j] * counts[j]) % mod_val
                
        total_length = 0
        for count in next_counts:
            total_length = (total_length + count) % mod_val
            
        return total_length