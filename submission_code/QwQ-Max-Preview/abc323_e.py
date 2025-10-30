MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    T = list(map(int, input[idx:idx+N]))
    idx += N
    
    T1 = T[0]
    S_min = max(0, X - T1 + 1)
    S_max = X
    
    if S_min > S_max:
        print(0)
        return
    
    inv_N = pow(N, MOD - 2, MOD)
    max_sum = X
    dp = [0] * (max_sum + 1)
    dp[0] = 1
    
    for j in range(max_sum + 1):
        for t in T:
            new_j = j + t
            if new_j > max_sum:
                continue
            dp[new_j] = (dp[new_j] + dp[j] * inv_N) % MOD
    
    total = 0
    for s in range(S_min, S_max + 1):
        total = (total + dp[s]) % MOD
    
    ans = total * inv_N % MOD
    print(ans)

if __name__ == '__main__':
    main()