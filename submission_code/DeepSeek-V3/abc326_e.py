MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute the modular inverses of N
    inv_N = pow(N, MOD-2, MOD)
    
    # Initialize dp array
    dp = [0] * (N + 1)
    
    # Compute dp in reverse order
    for x in range(N-1, -1, -1):
        # Calculate the sum of dp[y] for y > x
        # Since dp[y] is already computed for y > x
        # We can use a prefix sum approach
        # Initialize sum_dp_y to 0
        sum_dp_y = 0
        for y in range(x+1, N+1):
            sum_dp_y = (sum_dp_y + dp[y]) % MOD
        # Calculate dp[x]
        dp[x] = (A[x] + sum_dp_y * inv_N) % MOD
    
    # The expected value is dp[0]
    expected_value = dp[0]
    print(expected_value)

if __name__ == "__main__":
    main()