def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # answer[k] will hold the count (mod mod) of arithmetic subsequences of length k.
    # Note that every single element is an arithmetic subsequence of length 1.
    answer = [0] * (N + 1)
    answer[1] = N

    # dp[j] will be a dictionary that maps a difference d to another dictionary.
    # In dp[j][d], the inner dictionary maps a subsequence length L (>=2) to the count
    # of arithmetic subsequences ending at index j that have difference d and have length L.
    dp = [dict() for _ in range(N)]
    
    # We iterate over all pairs (i, j) with i < j to form subsequences.
    for j in range(N):
        for i in range(j):
            d = A[j] - A[i]
            if d not in dp[j]:
                dp[j][d] = {}
            # Every pair (i, j) immediately forms an arithmetic subsequence of length 2.
            dp[j][d][2] = (dp[j][d].get(2, 0) + 1) % mod

            # Now, if there are any subsequences ending at i with the same difference d,
            # extend each of them by A[j] to get a longer arithmetic subsequence.
            if d in dp[i]:
                for L, cnt in dp[i][d].items():
                    newL = L + 1
                    dp[j][d][newL] = (dp[j][d].get(newL, 0) + cnt) % mod

    # Now accumulate the counts from our dp states into the answer.
    for j in range(N):
        for d in dp[j]:
            for L, cnt in dp[j][d].items():
                if L <= N:
                    answer[L] = (answer[L] + cnt) % mod

    # Print the results for k = 1, 2, ... , N
    sys.stdout.write(" ".join(str(answer[k]) for k in range(1, N + 1)))

if __name__ == '__main__':
    main()