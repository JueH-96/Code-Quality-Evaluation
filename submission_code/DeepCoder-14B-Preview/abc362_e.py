from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    A = list(map(int, data[1:n+1]))
    mod = 998244353

    # Initialize DP table
    dp = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]

    for i in range(n):
        for j in range(i):
            current_diff = A[i] - A[j]
            # Add the new pair (j, i) as a subsequence of length 2
            dp[i][current_diff][2] += 1
            # Extend all subsequences ending at j with the same difference
            for l in dp[j][current_diff]:
                new_l = l + 1
                dp[i][current_diff][new_l] += dp[j][current_diff][l]
                # Apply modulo to prevent integer overflow
                dp[i][current_diff][new_l] %= mod

    # Prepare the result
    result = [0] * (n + 1)  # result[1] to result[n]

    # Handle k=1
    result[1] = n % mod

    # Handle k=2
    if n >= 2:
        result[2] = (n * (n - 1) // 2) % mod
    else:
        result[2] = 0

    # Handle k >=3
    for k in range(3, n + 1):
        total = 0
        for i in range(n):
            for d in dp[i]:
                total += dp[i][d].get(k, 0)
        result[k] = total % mod

    # Prepare the output
    output = []
    for k in range(1, n + 1):
        output.append(str(result[k] % mod))
    
    print(' '.join(output))

if __name__ == "__main__":
    main()