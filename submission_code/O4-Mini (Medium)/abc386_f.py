import sys
import threading
def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    K = int(next(it))
    S = next(it)
    T = next(it)
    n = len(S)
    m = len(T)
    # Quick checks
    if S == T:
        print("Yes")
        return
    if abs(n - m) > K:
        print("No")
        return
    # If one is empty
    if n == 0:
        print("Yes" if m <= K else "No")
        return
    if m == 0:
        print("Yes" if n <= K else "No")
        return
    # We do banded DP: only |i-j| <= K
    INF = K + 1
    # dp_prev holds dp[0][j] for j in [prev_j_low..prev_j_high]
    prev_j_low = 0
    prev_j_high = min(m, K)
    # initialize dp_prev[j] = j
    dp_prev = list(range(prev_j_low, prev_j_high + 1))
    # references
    S_str = S
    T_str = T
    for i in range(1, n + 1):
        # new row is dp[i][j] for j in [j_low..j_high]
        j_low = 0 if i <= K else i - K
        end = i + K
        j_high = m if end >= m else end
        width = j_high - j_low + 1
        dp_curr = [INF] * width
        # precompute delta = prev_j_low - j_low
        delta = prev_j_low - j_low
        # local refs
        prev = dp_prev
        plen = prev_j_high - prev_j_low + 1
        s_char = S_str[i - 1]
        # iterate j by offset
        for off in range(width):
            # j = j_low + off
            best = INF
            # deletion: from dp[i-1][j] + 1
            idx_del = off + delta
            if 0 <= idx_del < plen:
                cost = prev[idx_del] + 1
                if cost < best:
                    best = cost
            # insertion: from dp[i][j-1] + 1
            if off > 0:
                cost = dp_curr[off - 1] + 1
                if cost < best:
                    best = cost
            # match or replace: from dp[i-1][j-1] + (S[i-1]!=T[j-1])
            idx_rep = off + delta - 1
            if 0 <= idx_rep < plen:
                # j-1 = j_low + off - 1
                # compare s_char vs T[j-1]
                # compute penalty
                if s_char == T_str[j_low + off - 1]:
                    cost = prev[idx_rep]
                else:
                    cost = prev[idx_rep] + 1
                if cost < best:
                    best = cost
            dp_curr[off] = best
        # move on
        dp_prev = dp_curr
        prev_j_low = j_low
        prev_j_high = j_high
    # final check dp[n][m]
    # m must lie in [prev_j_low..prev_j_high]
    if prev_j_low <= m <= prev_j_high:
        if dp_prev[m - prev_j_low] <= K:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()