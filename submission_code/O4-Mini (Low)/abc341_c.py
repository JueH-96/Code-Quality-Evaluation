def main():
    import sys
    input = sys.stdin.readline

    H, W, N = map(int, input().split())
    T = input().strip()
    S = [input().rstrip() for _ in range(H)]

    # Precompute a bitmask for each row: 1 for land ('.'), 0 for sea ('#').
    land = [0] * H
    for i in range(H):
        m = 0
        row = S[i]
        # least significant bit corresponds to column 0
        for j in range(W - 1, -1, -1):
            m = (m << 1) | (1 if row[j] == '.' else 0)
        land[i] = m

    # dp[r] is a bitmask of possible positions in row r after processing so far.
    # Initially any land cell could be the crash site.
    dp = land[:]

    for c in T:
        newdp = [0] * H
        if c == 'L':
            # move left: (r,j) <- (r,j+1) in previous
            for i in range(H):
                # shift right one bit
                newdp[i] = (dp[i] >> 1) & land[i]
        elif c == 'R':
            # move right: (r,j) <- (r,j-1)
            # shift left
            mask = (1 << W) - 1
            for i in range(H):
                newdp[i] = ((dp[i] << 1) & mask) & land[i]
        elif c == 'U':
            # move up: (r,j) <- (r+1,j)
            for i in range(H - 1):
                newdp[i] = dp[i + 1] & land[i]
            # newdp[H-1] stays 0
        elif c == 'D':
            # move down: (r,j) <- (r-1,j)
            for i in range(1, H):
                newdp[i] = dp[i - 1] & land[i]
        dp = newdp

    # Count bits in dp
    ans = 0
    for m in dp:
        ans += m.bit_count()   # Python 3.8+: int.bit_count()
    print(ans)


if __name__ == "__main__":
    main()