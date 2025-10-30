# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    mod = 998244353

    def count_divisors(x):
        count = 0
        i = 1
        while i * i <= x:
            if x % i == 0:
                count += 1
                if i * i != x:
                    count += 1
            i += 1
        return count

    ans = 0
    
    if n <= 30:
        for length in range(1, int(n) + 1):
            for i in range(m**length):
                prod = 1
                temp = i
                for _ in range(length):
                    prod = (prod * (temp % m + 1)) 
                    temp //= m
                ans = (ans + count_divisors(prod)) % mod
        print(ans)
        return

    dp = [[0] * 128 for _ in range(17)]
    for i in range(1, m + 1):
        dp[1][i] = count_divisors(i)

    for i in range(2, 65):
        for j in range(1, 128):
            for k in range(1, m + 1):
                dp[i][j * k] = (dp[i][j * k] + dp[i-1][j]) % mod

    
    ans_vec = [0] * 128
    ans_vec[1] = 1
    mat = [[0] * 128 for _ in range(128)]
    for i in range(1, 128):
        for j in range(1, m + 1):
            mat[i * j % 128][i] = (mat[i * j % 128][i] + 1) % mod
    
    def mat_mul(a, b):
        c = [[0] * 128 for _ in range(128)]
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % mod
        return c

    def mat_pow(a, n):
        res = [[0] * 128 for _ in range(128)]
        for i in range(128):
            res[i][i] = 1
        while n > 0:
            if n % 2 == 1:
                res = mat_mul(res, a)
            a = mat_mul(a, a)
            n //= 2
        return res
    
    mat = mat_pow(mat, n)
    
    res_vec = [0] * 128
    for i in range(128):
        for j in range(128):
            res_vec[i] = (res_vec[i] + mat[i][j] * ans_vec[j]) % mod
    
    ans = 0
    for i in range(128):
        ans = (ans + res_vec[i] * dp[64][i]) % mod
    
    print(ans)

solve()