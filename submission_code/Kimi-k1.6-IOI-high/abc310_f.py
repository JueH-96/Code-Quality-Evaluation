MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    dp = [0] * (1 << 9)
    dp[0] = 1  # Initial state: no sums
    
    for a in A:
        inv_a = pow(a, MOD-2, MOD)
        new_dp = [0] * (1 << 9)
        for state in range(1 << 9):
            current_prob = dp[state]
            if current_prob == 0:
                continue
            
            # Compute forbidden set F_S
            F = {10}
            for s in range(1, 10):
                if (state >> (s-1)) & 1:
                    F.add(10 - s)
            
            max_small = min(a, 9)
            small_x = []
            for x in range(1, max_small + 1):
                if x not in F:
                    small_x.append(x)
            count_small = len(small_x)
            count_large = max(0, a - 10)
            valid_count = count_small + count_large
            
            if valid_count == 0:
                continue
            
            # Process small x's
            for x in small_x:
                new_state = state
                # Add x to the state
                new_state |= 1 << (x - 1)
                # Add s + x for each s in current state's sums
                for s in range(1, 10):
                    if (state >> (s - 1)) & 1:
                        if s + x < 10:
                            new_state |= 1 << (s + x - 1)
                new_dp[new_state] = (new_dp[new_state] + current_prob * inv_a) % MOD
            
            # Process large x's
            if count_large > 0:
                contrib = current_prob * count_large % MOD
                contrib = contrib * inv_a % MOD
                new_dp[state] = (new_dp[state] + contrib) % MOD
        
        dp = new_dp
    
    sum_dp = sum(dp) % MOD
    answer = (1 - sum_dp) % MOD
    print(answer)

if __name__ == '__main__':
    main()