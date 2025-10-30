def main():
    import sys
    from collections import defaultdict
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    N = int(data[0])
    A = list(map(int, data[1:]))

    # ans[k] will count the number of arithmetic subsequences of length k.
    # Every single element (length 1) is trivially an arithmetic sequence.
    ans = [0] * (N + 1)
    ans[1] = N  # Length one: each element of A is an arithmetic sequence.

    # We use dp[j] as a dictionary. For each index j, dp[j] maps a 
    # difference d (which can be any integer) to another dictionary.
    # This inner dictionary maps a subsequence length L (L>=2) to the number 
    # of arithmetic subsequences that end at index j with common difference d.
    dp = [dict() for _ in range(N)]
    
    # For each pair (i, j) with i < j, form a new arithmetic sequence (of length 2)
    # and try to extend any sequence ending at i having the same difference.
    for j in range(N):
        for i in range(j):
            d = A[j] - A[i]
            if d not in dp[j]:
                dp[j][d] = defaultdict(int)
            # Every new pair (i, j) forms a subsequence of length 2.
            dp[j][d][2] = (dp[j][d][2] + 1) % mod
            # If there are any arithmetic subsequences ending at i with difference d,
            # extend them by A[j] to form sequences of length L+1.
            if d in dp[i]:
                for L, count in dp[i][d].items():
                    dp[j][d][L+1] = (dp[j][d][L+1] + count) % mod

    # Collect the counts from dp into the ans array.
    # Note: dp only stores sequences of length >= 2.
    for j in range(N):
        for d, length_dict in dp[j].items():
            for L, count in length_dict.items():
                ans[L] = (ans[L] + count) % mod

    # Print the answer for each k = 1, 2, ..., N.
    # All subsequences not present in ans (for k >= 4 when none exist, etc.) remain 0.
    sys.stdout.write(" ".join(str(ans[k]) for k in range(1, N+1)))

if __name__ == '__main__':
    main()