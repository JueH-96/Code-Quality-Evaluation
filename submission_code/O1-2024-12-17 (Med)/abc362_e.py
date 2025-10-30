def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # Edge case: if N = 1, the answer is just 1 for k=1
    # (and 0 for any k>1, but there's only k=1 in that range).
    if N == 1:
        print(1)
        return

    # dp[i][diff][k] = number of arithmetic subsequences of length k
    # ending at index i with common difference = diff
    from collections import defaultdict
    dp = [defaultdict(lambda: [0] * (N+1)) for _ in range(N)]

    # For k = 1, every single element is trivially arithmetic.
    # We'll compute for k >= 2 using the above dp structure.
    for i in range(N):
        for j in range(i):
            diff = A[i] - A[j]
            # Account for any subsequence of length 2 formed by (j, i)
            dp[i][diff][2] = (dp[i][diff][2] + 1) % MOD
            # Extend subsequences of length >= 2 that end at j
            # and have the same difference
            for length in range(3, N+1):
                dp[i][diff][length] = (dp[i][diff][length] + dp[j][diff][length-1]) % MOD

    # Gather results
    # res[k] = total number of arithmetic subsequences of length k
    res = [0] * (N+1)

    # k=1: all single-element subsequences
    res[1] = N

    # Sum up dp for k >= 2
    for i in range(N):
        for diff_dict in dp[i].values():
            for length in range(2, N+1):
                res[length] = (res[length] + diff_dict[length]) % MOD

    # Print results for k = 1..N
    print(" ".join(str(res[k] % MOD) for k in range(1, N+1)))

# Do not forget to call main
if __name__ == "__main__":
    main()