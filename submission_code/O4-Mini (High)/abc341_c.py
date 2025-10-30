import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    N = int(next(it))
    T = next(it)
    # build bitmask for land cells in each row
    row_mask = []
    for _ in range(H):
        s = next(it)
        m = 0
        for j, ch in enumerate(s):
            if ch == '.':
                m |= 1 << j
        row_mask.append(m)
    # dp[r] will be a bitmask of possible columns in row r
    dp = row_mask.copy()
    # simulate each move
    for c in T:
        old = dp
        dp = [0] * H
        if c == 'L':
            # move left: new_col = old_col - 1
            for i in range(H):
                dp[i] = (old[i] >> 1) & row_mask[i]
        elif c == 'R':
            # move right: new_col = old_col + 1
            for i in range(H):
                dp[i] = (old[i] << 1) & row_mask[i]
        elif c == 'U':
            # move up: new_row = old_row - 1
            for i in range(H-1):
                dp[i] = old[i+1] & row_mask[i]
            # dp[H-1] stays 0
        elif c == 'D':
            # move down: new_row = old_row + 1
            for i in range(1, H):
                dp[i] = old[i-1] & row_mask[i]
            # dp[0] stays 0
    # count total bits set in dp
    ans = 0
    for m in dp:
        ans += m.bit_count()
    print(ans)

if __name__ == "__main__":
    main()