def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # If length difference exceeds K, it's impossible
    if abs(len(S) - len(T)) > K:
        print("No")
        return

    # Banded DP for edit distance (Levenshtein), in O(K*(|S|+|T|)) time.
    # We only need to know if distance <= K.

    n, m = len(S), len(T)
    INF = float('inf')

    # dp_prev will hold dp[i-1][*] band, dp_cur will hold dp[i][*] band
    dp_prev = []
    left_prev = 0
    right_prev = -1  # so that length = 0 initially

    for i in range(n+1):
        # The relevant j-range for dp[i]
        left_j = max(0, i - K)
        right_j = min(m, i + K)
        dp_cur = [INF] * (right_j - left_j + 1)

        for j in range(left_j, right_j+1):
            idx = j - left_j  # index in dp_cur
            if i == 0:
                # Base case: dp[0][j] = j (all inserts)
                dp_cur[idx] = j
            elif j == 0:
                # Base case: dp[i][0] = i (all deletes)
                dp_cur[idx] = i
            else:
                # Possible transitions:
                # 1) Delete last char from S: dp[i-1][j] + 1
                # 2) Insert last char of T:  dp[i][j-1] + 1
                # 3) Replace (or match) the last chars: dp[i-1][j-1] + cost
                cost = 0 if S[i-1] == T[j-1] else 1

                # dp[i-1][j]
                del_val = INF
                del_idx = j - left_prev
                if left_prev <= j <= right_prev and 0 <= del_idx < len(dp_prev):
                    del_val = dp_prev[del_idx] + 1

                # dp[i][j-1]
                ins_val = INF
                if j - 1 >= left_j:  # check it's in range of dp_cur
                    ins_val = dp_cur[idx - 1] + 1

                # dp[i-1][j-1]
                rep_val = INF
                rep_idx = (j - 1) - left_prev
                if left_prev <= j - 1 <= right_prev and 0 <= rep_idx < len(dp_prev):
                    rep_val = dp_prev[rep_idx] + cost

                dp_cur[idx] = min(del_val, ins_val, rep_val)

        dp_prev = dp_cur
        left_prev, right_prev = left_j, right_j

    # dp[n][m] is dp_prev[m - left_prev] if m is in [left_prev..right_prev]
    dist = dp_prev[m - left_prev]
    print("Yes" if dist <= K else "No")

# Do not forget to call main()!
if __name__ == "__main__":
    main()