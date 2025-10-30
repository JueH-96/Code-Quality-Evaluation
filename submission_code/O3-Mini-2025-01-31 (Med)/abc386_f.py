def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # Parse inputs
    K = int(data[0].strip())
    s = data[1].strip()
    t = data[2].strip()
    n = len(s)
    m = len(t)
    
    # Quick check: if the length difference is more than K, it is impossible
    if abs(n - m) > K:
        sys.stdout.write("No")
        return

    # We use Ukkonen's algorithm to compute the edit distance within a band of width 2*K+1.
    # dp[d] will store, for a given diagonal k = index - offset,
    # the farthest (largest) x-index in s that can be reached with d operations.
    # Here, index = k + offset; we set offset = K.
    
    offset = K
    # dp list covering diagonals from -K to K (2*K+1 entries)
    dp = [-1] * (2 * K + 1)
    dp[offset] = 0  # diagonal k = 0 gives starting x index 0

    # Try using at most K operations.
    for d in range(0, K + 1):
        # For each possible diagonal k in [-d, d] in steps of 2.
        for k in range(-d, d + 1, 2):
            idx = k + offset
            # Determine the starting x coordinate.
            if k == -d:
                # We can only bring our path from k+1 (i.e. an insertion into s).
                x = dp[idx + 1]
            elif k == d:
                # We can only get here from diagonal k-1 (i.e. a deletion from s)
                x = dp[idx - 1] + 1
            else:
                # We choose the better move: insertion or deletion? 
                # If dp[k-1] + 1 (deletion) is better than dp[k+1] (insertion), use that.
                if dp[idx - 1] + 1 > dp[idx + 1]:
                    x = dp[idx - 1] + 1
                else:
                    x = dp[idx + 1]
            y = x - k  # because on diagonal k, we have k = x - y
            # "Slide" along the diagonal as long as characters match.
            while x < n and y < m and s[x] == t[y]:
                x += 1
                y += 1
            dp[idx] = x
            
            # If we have reached the end of both strings, then d operations suffice.
            if x >= n and y >= m:
                sys.stdout.write("Yes")
                return

    # If we haven’t reached a full match within K operations, output "No".
    sys.stdout.write("No")
    
if __name__ == '__main__':
    main()