MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    B = list(map(int, input[idx:idx+N]))
    
    # Precompute suffix counts of -1
    suffix_q = [0] * (N + 1)
    count = 0
    for i in range(N-1, -1, -1):
        suffix_q[i] = count
        if B[i] == -1:
            count += 1
    
    # Precompute powers of M
    pow_cache = [1] * (N + 2)
    for i in range(1, N+2):
        pow_cache[i] = (pow_cache[i-1] * M) % MOD
    
    # Initialize pre_sum_prev
    pre_sum_prev = [0] * (M + 2)  # indexes 0 to M+1
    for x in range(M, 0, -1):
        pre_sum_prev[x] = 1
    
    ans = 0
    
    for i in range(1, N+1):
        current_B = B[i-1]
        dp_curr = [0] * (M + 2)  # indexes 0 to M+1
        
        if current_B == -1:
            for x in range(1, M+1):
                dp_curr[x] = pre_sum_prev[x+1] % MOD
        else:
            x = current_B
            if 1 <= x <= M:
                dp_curr[x] = pre_sum_prev[x+1] % MOD
        
        # Compute curr_sum
        curr_sum = [0] * (M + 2)
        for x in range(M, 0, -1):
            curr_sum[x] = (curr_sum[x+1] + dp_curr[x]) % MOD
        
        # Calculate contribution
        num_suf_free = suffix_q[i-1]
        Pi = pow_cache[num_suf_free]
        total = sum(dp_curr[1:M+1]) % MOD
        ans = (ans + total * Pi) % MOD
        
        # Update pre_sum_prev
        pre_sum_prev = curr_sum
    
    print(ans % MOD)

if __name__ == '__main__':
    main()