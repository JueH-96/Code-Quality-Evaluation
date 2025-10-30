def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    
    # dp[i] will track how many ways (capped at 2) we can form B[0:i] as a subsequence of A so far.
    # We only care whether there are at least 2 ways.
    dp = [0] * (M + 1)
    dp[0] = 1  # one way to form empty subsequence
    
    for a in A:
        # Traverse backwards so that every element is only used once per iteration.
        # For each valid position i, if B[i] matches the current element a,
        # then we update dp[i+1] by adding dp[i] (with capping to 2).
        for i in range(M - 1, -1, -1):
            if B[i] == a:
                dp[i+1] = min(2, dp[i+1] + dp[i])
        # Early exit if we already found at least two matches for full B.
        if dp[M] >= 2:
            sys.stdout.write("Yes")
            return
    sys.stdout.write("Yes" if dp[M] >= 2 else "No")

if __name__ == '__main__':
    main()