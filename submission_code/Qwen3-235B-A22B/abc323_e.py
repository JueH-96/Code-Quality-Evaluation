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
    R = X
    if L > R:
        print(0)
        return
    
    inv_N = pow(N, MOD-2, MOD)
    max_s = X
    dp = [0] * (max_s + 1)
    dp[0] = 1 % MOD
    
    for s in range(1, max_s + 1):
        total = 0
        for t in T:
            if t <= s:
                total += dp[s - t]
                if total >= MOD:
                    total -= MOD
        dp[s] = (total * inv_N) % MOD
    
    sum_ans = 0
    for s in range(L, R + 1):
        if s <= max_s:
            sum_ans = (sum_ans + dp[s]) % MOD
    
    ans = (sum_ans * inv_N) % MOD
    print(ans)

if __name__ == "__main__":
    main()