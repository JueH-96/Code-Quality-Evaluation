def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    K = int(input_data[0])
    S = input_data[1].strip()
    T = input_data[2].strip()
    n = len(S)
    m = len(T)
    
    # Quick check: if abs(n-m) > K, it's impossible.
    if abs(n - m) > K:
        sys.stdout.write("No")
        return

    # We'll use a banded dynamic programming approach.
    # Let dp[i][j] be minimal operations to convert S[:i] to T[:j],
    # but because K is small, we only care when difference j-i is within [-K, K].
    # We'll use 0-indexed lengths, dp dimension = (n+1) x (m+1)
    
    # Initialize dp[0][j] = j for j from 0 to m.
    # We only need to store two rows: prev and cur.
    INF = K + 1  # any value > K is not needed because if it exceeds K, it is effectively impossible.
    prev = [INF] * (m + 1)
    cur = [INF] * (m + 1)
    
    # For row 0:
    for j in range(m + 1):
        # Only store if j is within band of 0 (i.e., j <= K)
        if j <= K:
            prev[j] = j
    
    # Process rows i = 1 .. n
    for i in range(1, n + 1):
        # Compute range for j: j in [max(0, i-K), min(m, i+K)]
        low = max(0, i - K)
        high = min(m, i + K)
        # Set cur[j] = INF for j outside the band.
        for j in range(low, high + 1):
            # We are computing dp[i][j]
            # Cases:
            # 1. If j > 0: deletion from T side: dp[i][j] = dp[i][j-1] + 1
            left = cur[j-1] + 1 if j > low else INF
            # 2. If i > 0: deletion from S side (or insertion in T): dp[i][j] = prev[j] + 1
            up = prev[j] + 1
            # 3. If i,j >0: substitution or matching: dp[i][j] = prev[j-1] + (0 if S[i-1]==T[j-1] else 1)
            diag = INF
            if j > low and j - 1 >= max(0, (i - 1) - K) and j - 1 <= min(m, (i - 1) + K):
                diag = prev[j-1] + (0 if S[i-1] == T[j-1] else 1)
            # For the case when j == 0: then only possible is deletion from S side.
            if j == 0:
                cur[j] = up
            else:
                cur[j] = min(left, up, diag)
        # After finishing row i, swap prev and cur.
        prev, cur = cur, [INF] * (m + 1)
    
    if prev[m] <= K:
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

if __name__ == "__main__":
    main()