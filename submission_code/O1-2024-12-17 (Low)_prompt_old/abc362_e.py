def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # Edge case: If N=1, the answer is just 1 (only one subsequence of length 1)
    if N == 1:
        print(1)
        return

    # dp[i] will be a dictionary: dp[i][d] = another dict where dp[i][d][l] = number of arithmetic subsequences
    # ending at index i with common difference d and length = l.
    from collections import defaultdict
    dp = [defaultdict(lambda: defaultdict(int)) for _ in range(N)]

    # We will count how many subsequences of each length (1..N) exist.
    # For length 1, it's simply N (each single element is an arithmetic subsequence).
    result = [0]*(N+1)
    result[1] = N

    for i in range(N):
        for j in range(i):
            d = A[i] - A[j]
            # Extend all subsequences that end at j with difference d
            # If dp[j][d] has some counts for various lengths l,
            # then those can be extended to length l+1 when we pick i.
            for l, cnt in dp[j][d].items():
                dp[i][d][l+1] = (dp[i][d][l+1] + cnt) % MOD

            # Additionally, any pair (j, i) forms a new arithmetic subsequence of length=2
            dp[i][d][2] = (dp[i][d][2] + 1) % MOD

    # Accumulate counts into result array
    for i in range(N):
        for d in dp[i]:
            for l, cnt in dp[i][d].items():
                result[l] = (result[l] + cnt) % MOD

    # Print results for k=1..N
    print(" ".join(str(result[k] % MOD) for k in range(1, N+1)))

# Call solve() after defining
if __name__ == "__main__":
    solve()