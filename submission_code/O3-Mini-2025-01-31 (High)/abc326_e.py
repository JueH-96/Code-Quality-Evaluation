def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    
    # Read input values.
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    
    # Let dp[x] denote the expected additional salary if the current state is x.
    # From the problem, if current x and we roll y:
    #   if y <= x, the process terminates (gain 0 additional yen).
    #   if y > x, then we immediately get A_y yen and the process continues from state y.
    # Thus, for state x (0 <= x < N), we have:
    #   dp[x] = (1/N) * sum_{y=x+1}^{N} (A[y] + dp[y])
    # Note: dp[N] = 0 (when x equals N there is no available y > x).
    #
    # For x = 0 (the starting state), we want dp[0] = (1/N)*sum_{y=1}^{N} (A[y] + dp[y]).
    #
    # To compute the dp values efficiently, define:
    #   S(x) = sum_{y=x}^{N} (A[y] + dp[y])
    # Then for x < N, we have:
    #   dp[x] = S(x+1) / N
    # and
    #   S(x) = A[x] + dp[x] + S(x+1)
    # (Here, A[0] is not defined because when state=0 no payout is attached, 
    # so we only apply the recurrence for indices 1...N.)
    #
    # Thus, we compute for i from N downto 1:
    #   dp[i] = S(i+1) / N  with dp[N] = 0 since S(N+1)=0,
    #   S(i) = A[i] + dp[i] + S(i+1)
    #
    # Finally, dp[0] = (1/N)*S(1).
    
    # We'll compute in one variable "s" that holds S(i+1) as we iterate i = N, N-1, ..., 1.
    # For each i (using 1-indexing for A), let dp = s / N (in mod arithmetic, multiplied by inv_N)
    # and then update s = A[i] + dp + s.
    
    inv_N = pow(N, mod - 2, mod)  # Modular inverse of N.
    s = 0  # Initially, s = S(N+1) = 0.
    
    # Process states backwards: i from N down to 1.
    for i in range(N, 0, -1):
        dp = s * inv_N % mod
        s = (A[i - 1] + dp + s) % mod  # A[i-1] represents A_i (since list A is 0-indexed)
    
    # Finally, the expected salary starting from state 0 is dp(0) = S(1)/N.
    ans = s * inv_N % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()