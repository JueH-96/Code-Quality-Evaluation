MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    T = list(map(int, input[ptr:ptr+N]))
    ptr += N
    T1 = T[0]
    a = max(0, X - T1 + 1)
    max_s = X
    inv_N = pow(N, MOD-2, MOD)
    
    # Frequency array for T values <= max_s
    freq = [0] * (max_s + 1)
    for t in T:
        if t <= max_s:
            freq[t] += 1
    
    dp = [0] * (max_s + 1)
    dp[0] = 1
    total = 0
    
    for k in range(0, X + 1):
        # Add current sum for step k (m-1 =k)
        current_sum = sum(dp[a:]) % MOD
        total = (total + current_sum) % MOD
        
        if k == X:
            break
        
        new_dp = [0] * (max_s + 1)
        for s in range(max_s + 1):
            if dp[s] == 0:
                continue
            for t in T:
                if s + t > max_s:
                    continue
                new_dp[s + t] = (new_dp[s + t] + dp[s]) % MOD
        
        # Multiply by inv_N
        for s in range(max_s + 1):
            new_dp[s] = new_dp[s] * inv_N % MOD
        
        dp = new_dp
    
    print(total % MOD)

if __name__ == "__main__":
    main()