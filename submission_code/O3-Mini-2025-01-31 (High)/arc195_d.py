def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    t = int(data[0])
    pos = 1
    out_lines = []
    for _ in range(t):
        if pos >= len(data):
            break
        n = int(data[pos])
        pos += 1
        # read the sequence (we use 0-indexed list)
        A = list(map(int, data[pos:pos+n]))
        pos += n
        # dp[i] = min total cost to completely delete subarray A[i:n].
        dp = [0]*(n+1)
        dp[n] = 0  # empty sequence, no cost.
        # Precompute for each index i the "next_different" index:
        # nd[i] = smallest j > i with A[j] != A[i] (or n if none)
        nd = [n]*n
        if n:
            nd[n-1] = n
            for i in range(n-2, -1, -1):
                if A[i] == A[i+1]:
                    nd[i] = nd[i+1]
                else:
                    nd[i] = i+1
        # best dictionary: for each value v, we keep best[v] = min{ dp(j+1) + j } over already–processed indices j with A[j]==v.
        best = {}
        # Process i = n-1 downto 0.
        for i in range(n-1, -1, -1):
            # If the suffix A[i:] is homogeneous, we can clear it with one deletion.
            if nd[i] == n:
                dp[i] = 1
            else:
                dp[i] = 1 + dp[i+1]  # option: delete A[i] separately.
            v = A[i]
            if v in best:
                # candidate from “merging”: cost = (best[v] - i)
                candidate = best[v] - i
                if candidate < dp[i]:
                    dp[i] = candidate
            # Update best[v]: we want to set best[v] = min( existing best[v], dp[i+1] + i )
            new_val = dp[i+1] + i  if i < n else i
            if v in best:
                if new_val < best[v]:
                    best[v] = new_val
            else:
                best[v] = new_val
        out_lines.append(str(dp[0]))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()