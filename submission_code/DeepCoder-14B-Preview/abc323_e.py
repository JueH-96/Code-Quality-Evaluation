import math

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = int(input[1])
    T = list(map(int, input[2:2+N]))
    T1 = T[0]
    
    # Compute s_min
    s_min = (X + 0.5 - T1)
    s_min = math.floor(s_min) + 1
    s_min = max(0, s_min)
    
    inv_N = pow(N, MOD-2, MOD)
    
    # Initialize DP
    current_dp = [0] * (X + 1)
    current_dp[0] = 1
    
    total = 0
    
    while True:
        # Calculate sum_contribution for current_dp
        sum_contribution = 0
        for s in range(s_min, X + 1):
            sum_contribution = (sum_contribution + current_dp[s]) % MOD
        
        # Add to total
        total = (total + sum_contribution * inv_N) % MOD
        
        # Compute next_dp
        next_dp = [0] * (X + 1)
        for s in range(X + 1):
            if current_dp[s] == 0:
                continue
            contrib = (current_dp[s] * inv_N) % MOD
            for t in T:
                new_s = s + t
                if new_s > X:
                    continue
                next_dp[new_s] = (next_dp[new_s] + contrib) % MOD
        
        # Check if next_dp is all zero
        if all(v == 0 for v in next_dp):
            break
        
        current_dp = next_dp
    
    print(total % MOD)

if __name__ == '__main__':
    main()