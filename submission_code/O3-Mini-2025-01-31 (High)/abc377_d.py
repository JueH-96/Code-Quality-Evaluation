def main():
    import sys
    # Read and split input bytes (faster I/O)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Parse input: first two numbers are N and M
    N = int(data[0])
    M = int(data[1])
    
    # We will use 1-indexing for positions from 1 to M.
    # Let INF be a value larger than any R so that if no interval starts at a position, we use M+1.
    INF = M + 1

    # best[i] will store the smallest R among all intervals that start exactly at i.
    best = [INF] * (M + 2)
    
    # Read the N intervals. Each interval is (L, R).
    pos = 2
    for _ in range(N):
        L = int(data[pos])
        R = int(data[pos + 1])
        pos += 2
        # Update best[L] if this R is smaller.
        if R < best[L]:
            best[L] = R

    # Compute dp[x] = min{ best[j] for j >= x }.
    # This represents the smallest right endpoint among all intervals that start at or after x.
    dp = [INF] * (M + 2)
    dp[M + 1] = INF  # for compatibility; no index beyond M.
    for i in range(M, 0, -1):
        # dp[i] is the minimum of best[i] (intervals starting exactly at i) and dp[i+1] (for larger starting positions)
        dp[i] = best[i] if best[i] < dp[i + 1] else dp[i + 1]

    # For a candidate pair (l, r) with 1 <= l <= r <= M, the condition is:
    # "for every interval [L, R] it is not the case that (l <= L and r >= R)"
    # Notice that if an interval starts before l then l > L and the condition is automatically satisfied.
    # Only intervals with L >= l require r < R.
    # So for a fixed l, r must satisfy r < (minimum R over intervals with L >= l).
    # Let X = dp[l]. If no interval starts at or after l then dp[l] is INF so any r in [l, M] works.
    # Otherwise, we must have r <= dp[l] - 1.
    #
    # Hence, for each l, the valid r's are those in the range:
    #   r in [l, min(M, dp[l] - 1)]
    #
    # We accumulate the number of valid pairs for each l.
    
    ans = 0
    for l in range(1, M + 1):
        # r_max is the maximum r allowed for fixed l.
        r_max = dp[l] - 1
        if r_max > M:
            r_max = M
        # Only count if the range is non-empty.
        if r_max >= l:
            ans += (r_max - l + 1)
    
    # Write out the final answer.
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()