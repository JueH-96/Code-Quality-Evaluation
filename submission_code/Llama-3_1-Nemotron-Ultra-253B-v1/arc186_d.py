MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # dp[tight][c] represents the number of ways for the current state
    dp_tight = [ [0]*(N+2) for _ in range(2) ]
    # Initialize for i = N
    dp_tight[0][0] = 1
    dp_tight[1][0] = 1
    
    for i in reversed(range(N)):
        new_dp = [ [0]*(N+2) for _ in range(2) ]
        max_k_not_tight = N - i - 1
        for tight in [0, 1]:
            current_dp = dp_tight[tight]
            # Precompute prefix sums
            prefix = [0]*(N+2)
            for c in range(N+1):
                prefix[c+1] = (prefix[c] + current_dp[c]) % MOD
            for c in range(N+1):
                if current_dp[c] == 0:
                    continue
                # For each possible c, compute the valid k and accumulate
                if N - i == 0:
                    # Not possible since i starts from N-1
                    continue
                remaining_length = N - i - 1
                if remaining_length == 0:
                    # Must choose k=0
                    if c == 0:
                        k = 0
                        new_tight_flag = tight and (k == A[i])
                        new_c = c - 1 + k
                        if new_c >= 0 and new_c <= N:
                            new_dp[new_tight_flag][new_c] = (new_dp[new_tight_flag][new_c] + current_dp[c]) % MOD
                    continue
                # Determine the valid k range
                if tight:
                    max_k = min(A[i], remaining_length)
                else:
                    max_k = remaining_length
                min_k = 1
                if max_k < min_k:
                    continue
                # Split into k < A[i] (new_tight=0) and k == A[i] (new_tight=1)
                if tight:
                    upper = A[i] - 1
                    if upper >= min_k:
                        lower_c = c - 1 + 1
                        upper_c = c - 1 + min(upper, max_k)
                        if lower_c > upper_c:
                            sum_loose = 0
                        else:
                            sum_loose = (prefix[upper_c + 1] - prefix[lower_c]) % MOD
                        new_dp[0][c] = (new_dp[0][c] + sum_loose) % MOD
                    if A[i] <= max_k:
                        k = A[i]
                        new_c = c - 1 + k
                        if new_c >= 0 and new_c <= N:
                            sum_tight = current_dp[c - 1 + k] if (c - 1 + k) <= N else 0
                            new_dp[1][c] = (new_dp[1][c] + sum_tight) % MOD
                else:
                    lower_c = c - 1 + 1
                    upper_c = c - 1 + max_k
                    if lower_c > upper_c:
                        sum_loose = 0
                    else:
                        sum_loose = (prefix[upper_c + 1] - prefix[lower_c]) % MOD
                    new_dp[0][c] = (new_dp[0][c] + sum_loose) % MOD
        dp_tight = new_dp
    
    # The answer is the sum of dp[0][1][1] and dp[0][1][0] if tight is not required
    # Initially, we have to split into 1 sequence, and the tight flag is 1 (since we start with empty prefix)
    print(dp_tight[1][1] % MOD)

if __name__ == '__main__':
    main()