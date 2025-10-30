def main():
    import sys

    data = sys.stdin.read().strip().split()
    K = int(data[0])
    S = data[1]
    T = data[2]

    # If the length difference is already greater than K,
    # we cannot fix that with <= K insert/delete operations.
    if abs(len(S) - len(T)) > K:
        print("No")
        return

    # Quick check: if S == T, distance is 0 -> always yes.
    if S == T:
        print("Yes")
        return

    n = len(S)
    m = len(T)

    # We use a "banded" dynamic programming approach (sometimes called "Ukkonen's Algorithm" for
    # bounding the edit distance) to compute whether edit distance <= K in O(K * (n + m)).
    # dp[i][j] = edit distance between S[:i] and T[:j].
    # We only keep track of columns j in [i-K, i+K], so each row has at most 2K+1 entries.

    INF = K + 1  # Any cost above K can be treated as "infinite" for our purposes

    # dp_prev will hold the previous row's DP values within the band
    # old_j0, old_j1 will track the range of j that was valid in the previous row
    dp_prev = []
    old_j0 = 0
    old_j1 = -1

    # Iterate over i from 0..n (where dp[i][*] = cost of converting S[:i] to T[:j])
    # i.e. i is length of prefix of S we are converting.
    for i in range(n + 1):
        # Determine new band range for j
        new_j0 = max(0, i - K)
        new_j1 = min(m, i + K)

        width = new_j1 - new_j0 + 1
        dp_curr = [INF] * width  # This row's DP array within the band

        if i == 0:
            # Base case: dp[0][j] = j (cost of inserting j chars to convert "" to T[:j])
            for j in range(new_j0, new_j1 + 1):
                dp_curr[j - new_j0] = j
        else:
            # Fill dp[i][j] for j in [new_j0..new_j1]
            for j in range(new_j0, new_j1 + 1):
                j_idx = j - new_j0
                res = INF

                # If in range for dp[i-1][j], that means we can do a deletion from S
                if old_j0 <= j <= old_j1:
                    res = min(res, dp_prev[j - old_j0] + 1)

                # If j > new_j0, we can do an insertion (dp[i][j-1] + 1)
                if j > new_j0:
                    res = min(res, dp_curr[j_idx - 1] + 1)

                # If we can do substitution (dp[i-1][j-1] + cost), j>0 for that
                if j > 0 and (old_j0 <= j - 1 <= old_j1):
                    cost = 0 if S[i - 1] == T[j - 1] else 1
                    res = min(res, dp_prev[(j - 1) - old_j0] + cost)

                dp_curr[j_idx] = res

        dp_prev = dp_curr
        old_j0, old_j1 = new_j0, new_j1

    # After filling up dp for i=n, dp[n][m] should be in dp_prev if m is in [old_j0..old_j1]
    if old_j0 <= m <= old_j1:
        if dp_prev[m - old_j0] <= K:
            print("Yes")
        else:
            print("No")
    else:
        print("No")


if __name__ == "__main__":
    main()