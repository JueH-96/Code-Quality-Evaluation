MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if N == 0:
        print(0)
        return
    
    max_a_prev = N  # Initial maximum possible a_prev is N-1 - 0 = N-1, but A[0] may be smaller
    prev_tight = [0] * (N + 2)
    prev_loose = [0] * (N + 2)
    
    # Initialize for i=0
    max_a_0 = A[0]
    for a in range(0, max_a_0 + 1):
        if a < A[0]:
            prev_loose[a] = 1
        else:
            prev_tight[a] = 1
    
    for i in range(1, N):
        curr_tight = [0] * (N + 2)
        curr_loose = [0] * (N + 2)
        prefix_loose = [0] * (N + 2)
        prefix_tight = [0] * (N + 2)
        
        # Build prefix sums
        for j in range(N + 2):
            if j > 0:
                prefix_loose[j] = (prefix_loose[j-1] + prev_loose[j-1]) % MOD
                prefix_tight[j] = (prefix_tight[j-1] + prev_tight[j-1]) % MOD
        
        max_a_i = N - i - 1
        # a_i can be up to max_a_i = (N-1 -i)
        for a_i in range(0, max_a_i + 1 + 1):  # up to max_a_i inclusive
            # Compute curr_tight[a_i]
            a_prev = 1 + a_i - A[i]
            if a_prev >= 0:
                if a_prev <= N - (i-1) -1:  # a_prev <= M_{i-1}
                    if a_prev >= 0:
                        if a_prev <= N:
                            curr_tight[a_i] = prev_tight[a_prev]
            curr_tight[a_i] %= MOD
            
            # Compute curr_loose[a_i]
            # Contribution from prev_loose: a_prev <= a_i + 1
            upper_lo = a_i + 1
            if upper_lo >= 0:
                sum_lo = prefix_loose[upper_lo + 1] if (upper_lo + 1) < (N+2) else prefix_loose[N+1]
            else:
                sum_lo = 0
            sum_lo %= MOD
            
            # Contribution from prev_tight where S[i] < A[i]
            lower = a_i - A[i] + 2
            upper = a_i + 1
            lower = max(lower, 0)
            sum_ti = 0
            if lower <= upper:
                if upper >= 0:
                    right = min(upper + 1, N+1)
                    left = min(lower, N+1)
                    sum_ti = (prefix_tight[right] - prefix_tight[left]) % MOD
            sum_ti %= MOD
            
            curr_loose[a_i] = (sum_lo + sum_ti) % MOD
        
        prev_tight, prev_loose = curr_tight, curr_loose
    
    # After processing all N elements, a_{N-1} must be 0
    # The current arrays are for i=N-1, a_i must be 0
    ans = (prev_tight[0] + prev_loose[0]) % MOD
    print(ans)

if __name__ == "__main__":
    main()