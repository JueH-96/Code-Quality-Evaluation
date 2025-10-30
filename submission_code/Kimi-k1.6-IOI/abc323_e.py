MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = int(data[1])
    T = list(map(int, data[2:2+N]))
    
    T1 = T[0]
    L = max(0, X - T1 + 1)
    inv_N = pow(N, MOD-2, MOD)
    
    max_t = X
    dp = [0] * (max_t + 1)
    dp[0] = 1
    
    for t in range(1, max_t + 1):
        for Ti in T:
            prev = t - Ti
            if prev >= 0:
                dp[t] = (dp[t] + dp[prev] * inv_N) % MOD
    
    result = 0
    for s in range(L, max_t + 1):
        result = (result + dp[s] * inv_N) % MOD
    
    print(result)

if __name__ == "__main__":
    main()