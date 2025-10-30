class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        mod = 10**9 + 7

        def multiply(a, b):
            c = [[0, 0], [0, 0]]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % mod
            return c

        def power(a, k):
            res = [[1, 0], [0, 1]]
            while k > 0:
                if k % 2 == 1:
                    res = multiply(res, a)
                a = multiply(a, a)
                k //= 2
            return res

        same = 0
        for i in range(n):
            if s == t:
                same += 1
            s = s[1:] + s[0]

        if same == 0:
            return 0

        if s == t:
            mat = [[same - 1, same], [n - same, n - same - 1]]
            res = power(mat, k)
            return res[0][0]
        else:
            mat = [[same - 1, same], [n - same, n - same - 1]]
            res = power(mat, k)
            return res[0][1]