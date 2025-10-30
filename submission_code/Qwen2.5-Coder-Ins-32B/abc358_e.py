# YOUR CODE HERE
MOD = 998244353

def modinv(x, mod):
    return pow(x, mod - 2, mod)

def comb(n, k, mod):
    if k > n:
        return 0
    num = 1
    den = 1
    for i in range(k):
        num = num * (n - i) % mod
        den = den * (i + 1) % mod
    return num * modinv(den, mod) % mod

def main():
    import sys
    input = sys.stdin.read().split()
    K = int(input[0])
    C = list(map(int, input[1:]))
    
    dp = [[0] * (K + 1) for _ in range(27)]
    dp[0][0] = 1
    
    for i in range(1, 27):
        for j in range(K + 1):
            for k in range(C[i - 1] + 1):
                if j + k <= K:
                    dp[i][j + k] = (dp[i][j + k] + dp[i - 1][j] * comb(j + k, k, MOD)) % MOD
    
    result = sum(dp[26][1:]) % MOD
    print(result)

if __name__ == "__main__":
    main()