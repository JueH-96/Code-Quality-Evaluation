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
        # But since we are computing in reverse order, we can accumulate the sum
        if x == N-1:
            total = 0
        else:
            total = dp[x+1]
        # The expected value is (A_{x+1} + total) * inv_N
        if x < N:
            dp[x] = (A[x] + total) * inv_N % MOD
        else:
            dp[x] = 0
    
    # The answer is dp[0]
    print(dp[0])

if __name__ == "__main__":
    main()