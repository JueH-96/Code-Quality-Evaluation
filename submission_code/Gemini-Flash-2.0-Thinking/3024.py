class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        MOD = 10**9 + 7

        def multiply(A, B):
            C = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    for l in range(n):
                        C[i][j] = (C[i][j] + A[i][l] * B[l][j]) % MOD
            return C

        def power(matrix, k):
            result = [[0] * n for _ in range(n)]
            for i in range(n):
                result[i][i] = 1
            base = [row[:] for row in matrix]
            while k > 0:
                if k % 2 == 1:
                    result = multiply(result, base)
                base = multiply(base, base)
                k //= 2
            return result

        if s == t:
            initial_state = 0
        else:
            t_double = t + t
            try:
                initial_state = t_double.index(s)
            except ValueError:
                return 0

        target_index = -1
        for i in range(n):
            shifted_s = s[i:] + s[:i]
            if shifted_s == t:
                target_index = i
                break

        if target_index == -1:
            return 0

        transition_matrix = [[1] * n for _ in range(n)]
        for i in range(n):
            transition_matrix[i][i] = 0

        final_matrix = power(transition_matrix, k)

        return final_matrix[0][target_index]