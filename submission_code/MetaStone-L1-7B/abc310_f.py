MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    max_k = 10
    inv = [0] * (max_k + 1)
    for i in range(1, max_k + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    dp = [0] * (max_k + 1)
    dp[0] = 1  # Initially, sum 0 with probability 1
    
    for a in A:
        k = min(a, max_k)
        inv_k = inv[k]
        new_dp = [0] * (max_k + 1)
        
        for s in range(max_k + 1):
            if dp[s] == 0:
                continue
            
            # Add the case where the die is not taken
            new_dp[s] = (new_dp[s] + dp[s]) % MOD
            
            # Add the cases where the die is taken, for each possible value v
            for v in range(1, k + 1):
                new_s = s + v
                if new_s > max_k:
                    continue
                add = (dp[s] * inv_k) % MOD
                new_dp[new_s] = (new_dp[new_s] + add) % MOD
        
        dp = new_dp
    
    print(dp[10] % MOD)

if __name__ == "__main__":
    main()