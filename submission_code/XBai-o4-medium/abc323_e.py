MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    T = list(map(int, input[ptr:ptr + N]))
    ptr += N
    
    T1 = T[0]
    L = X - T1 + 1
    L = max(L, 0)
    R = X
    
    max_sum = X
    dp = [0] * (max_sum + 1)
    dp[0] = 1
    
    inv_N = pow(N, MOD - 2, MOD)
    
    for s in range(max_sum + 1):
        if dp[s] == 0:
            continue
        for t in T:
            next_s = s + t
            if next_s > max_sum:
                continue
            dp[next_s] = (dp[next_s] + dp[s] * inv_N) % MOD
    
    sum_prob = 0
    for s in range(L, R + 1):
        if s > max_sum:
            continue
        sum_prob = (sum_prob + dp[s]) % MOD
    
    ans = sum_prob * inv_N % MOD
    print(ans)
    
if __name__ == '__main__':
    main()