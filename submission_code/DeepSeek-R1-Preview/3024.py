MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        if len(t) != n:
            return 0
        if s == t:
            same = True
        else:
            same = False
        s2 = s + s
        if t not in s2:
            return 0
        
        def minimal_period(s):
            n = len(s)
            pi = [0] * n
            for i in range(1, n):
                j = pi[i-1]
                while j > 0 and s[i] != s[j]:
                    j = pi[j-1]
                if s[i] == s[j]:
                    j += 1
                pi[i] = j
            d = n - pi[-1]
            if n % d == 0:
                return d
            else:
                return n
        
        d = minimal_period(s)
        
        def find_min_rotation(s, t):
            if s == t:
                return d
            s2 = s + s
            idx = s2.find(t)
            return idx
        
        r = find_min_rotation(s, t)
        r_mod = r % d
        
        cnt = [0] * d
        for l in range(1, n):
            cnt[l % d] += 1
        
        sum0 = cnt[0]
        sum_rest = sum(cnt[1:]) if d > 1 else 0
        
        if d == 1:
            if r_mod != 0:
                return 0
            return pow(n-1, k, MOD)
        
        M = [[sum0, sum_rest], [sum_rest, cnt[0]]]
        
        def matrix_pow(mat, power):
            result = [[1, 0], [0, 1]]
            while power > 0:
                if power % 2 == 1:
                    result = matrix_mult(result, mat)
                mat = matrix_mult(mat, mat)
                power //= 2
            return result
        
        def matrix_mult(a, b):
            res = [[0]*2 for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        res[i][j] = (res[i][j] + a[i][k] * b[k][j]) % MOD
            return res
        
        M_k = matrix_pow(M, k)
        
        a0 = 1 if r_mod == 0 else 0
        b0 = 0 if r_mod == 0 else 1
        
        a_k = (M_k[0][0] * a0 + M_k[0][1] * b0) % MOD
        b_k = (M_k[1][0] * a0 + M_k[1][1] * b0) % MOD
        
        if r_mod == 0:
            return a_k % MOD
        else:
            return b_k % MOD