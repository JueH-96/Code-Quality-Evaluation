def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, X, Y = map(int, input_data[:3])
    dishes = list(zip(
        map(int, input_data[3::2]),
        map(int, input_data[4::2])
    ))

    # We will use a DP approach where:
    #   dp[c][s] = the minimum total saltiness achievable using exactly c dishes
    #              with total sweetness = s.
    # Dimensions:
    #   c ranges from 0 to N (inclusive).
    #   s ranges from 0 to X (inclusive).
    #
    # Algorithm sketch:
    #  1) Initialize dp with "infinity" saltiness.
    #  2) dp[0][0] = 0 (0 dishes → sweetness=0, saltiness=0).
    #  3) For each dish (a, b), update dp in descending order of c and s:
    #       for c from current maximum possible down to 0:
    #         for s from X-a down to 0:
    #           if dp[c][s] != inf:
    #              dp[c+1][s+a] = min(dp[c+1][s+a], dp[c][s] + b)
    #
    #  4) After processing all dishes, find the maximum c such that
    #     there exists an s for which dp[c][s] <= Y.
    #
    # Complexity:
    #   We do N updates; for each dish we iterate c up to N and s up to X.
    #   So O(N^2 * X) = 80 * 80 * 10,000 = 64e6 in the worst case, which is large.
    #   In a fast language it can be borderline feasible. In Python, we must
    #   implement it carefully to pass. We'll try to keep it as optimized as 
    #   reasonably possible in Python.
    #
    #   (If this were a lower-level language, we could optimize it further. 
    #    But we'll attempt to be efficient in Python too.)

    INF = 10**15
    # dp[c][s] = minimal saltiness for c dishes, total sweetness = s
    # We'll build this as a list of lists.
    dp = [[INF]*(X+1) for _ in range(N+1)]
    dp[0][0] = 0

    for (a, b) in dishes:
        # Update dp in descending order of c and s
        # c can go up to N-1 because we add +1 dish next
        for c in range(N-1, -1, -1):
            # s from X-a down to 0 to avoid double counting in the same iteration
            for s in range(X - a, -1, -1):
                if dp[c][s] != INF:
                    salt_before = dp[c][s]
                    new_salt = salt_before + b
                    # new sweetness = s + a
                    # new number of dishes = c+1
                    if new_salt < dp[c+1][s+a]:
                        dp[c+1][s+a] = new_salt

    ans = 0
    # Find the maximum c such that dp[c][s] <= Y for some s
    for c in range(N, -1, -1):
        # Once we find a feasible c, we can stop
        feasible = False
        row = dp[c]
        for s in range(X+1):
            if row[s] <= Y:
                feasible = True
                break
        if feasible:
            ans = c
            break

    print(ans)

# Call main() at the end
if __name__ == "__main__":
    main()