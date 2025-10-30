import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    mod = 998244353
    H = int(next(it)); W = int(next(it))
    # Read the grid A, 1-based indexing
    A = [[0]*(W+1)]
    for _ in range(H):
        row = [0]*(W+1)
        for j in range(1, W+1):
            row[j] = int(next(it))
        A.append(row)
    # Build initial dp array, dp[i][j] = sum of products to (i,j)
    dp = [[0]*(W+1) for _ in range(H+1)]
    for i in range(1, H+1):
        for j in range(1, W+1):
            if i == 1 and j == 1:
                dp[i][j] = A[i][j] % mod
            else:
                s = dp[i-1][j] + dp[i][j-1]
                if s >= mod: s -= mod
                dp[i][j] = (A[i][j] * s) % mod
    # Read queries
    Q = int(next(it))
    sh = int(next(it)); sw = int(next(it))
    out = []
    for _ in range(Q):
        d = next(it); a = int(next(it))
        # move Takahashi
        if d == 'L':
            sw -= 1
        elif d == 'R':
            sw += 1
        elif d == 'U':
            sh -= 1
        else:  # 'D'
            sh += 1
        # update A[sh][sw]
        A[sh][sw] = a
        # propagate dp updates in the sub‐rectangle i>=sh, j>=sw
        # we do it in increasing order of i+j so dependencies are met
        start_sum = sh + sw
        end_sum = H + W
        for s in range(start_sum, end_sum+1):
            # i + j = s, and we need i>=sh, j>=sw
            # so i >= max(sh, s - W) and i <= min(H, s - sw)
            i_low = sh if sh > s - W else s - W
            i_high = H if H < s - sw else s - sw
            if i_low > i_high:
                continue
            for i in range(i_low, i_high+1):
                j = s - i
                # now re-compute dp[i][j]
                # parents dp[i-1][j] and dp[i][j-1] are already up-to-date
                val = dp[i-1][j] + dp[i][j-1]
                if val >= mod: val -= mod
                dp[i][j] = (A[i][j] * val) % mod
        # answer is dp[H][W]
        out.append(str(dp[H][W]))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()