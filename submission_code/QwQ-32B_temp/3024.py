class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if n != len(t):
            return 0
        
        X = self.find_rotations(s, t, n)
        if not X:
            return 0
        
        # Compute a_k and b_k using matrix exponentiation
        mat = [[0, n-1], [1, n-2]]
        mat_k = self.matrix_pow(mat, k, MOD)
        a_k = mat_k[0][0]
        b_k = mat_k[1][0]
        
        res = 0
        for x in X:
            if x == 0:
                res = (res + a_k) % MOD
            else:
                res = (res + b_k) % MOD
        return res % MOD
    
    def find_rotations(self, s, t, n):
        T = t + '#' + s + s
        Z = self.compute_z_array(T)
        start_s_part = len(t) + 1
        X = []
        for i in range(start_s_part, start_s_part + 2 * n):
            if Z[i] >= n:
                pos_in_ss = i - (len(t) + 1)
                if 0 <= pos_in_ss < n:
                    X.append(pos_in_ss)
        return X
    
    def compute_z_array(self, s):
        n = len(s)
        Z = [0] * n
        Z[0] = n
        l, r = 0, 0
        for i in range(1, n):
            if i > r:
                l = r = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                Z[i] = r - l
                r -= 1
            else:
                k = i - l
                if Z[k] < r - i + 1:
                    Z[i] = Z[k]
                else:
                    l = i
                    while r < n and s[r - l] == s[r]:
                        r += 1
                    Z[i] = r - l
                    r -= 1
        return Z
    
    def matrix_mult(self, A, B, mod):
        res = [[0]*2 for _ in range(2)]
        res[0][0] = (A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod
        res[0][1] = (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod
        res[1][0] = (A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod
        res[1][1] = (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod
        return res
    
    def matrix_pow(self, mat, power, mod):
        result = [[1, 0], [0, 1]]  # Identity matrix
        while power > 0:
            if power % 2 == 1:
                result = self.matrix_mult(result, mat, mod)
            mat = self.matrix_mult(mat, mat, mod)
            power //= 2
        return result