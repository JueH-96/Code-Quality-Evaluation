import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    M = 998244353
    data = sys.stdin.read().split()
    N = int(data[0]); K = int(data[1])
    S = data[2].strip()
    # dp[mask] = number of ways for prefix ending at i-1 with last up to K-1 chars encoded:
    # mask bit j = char at position (i-1-j), for j=0..K-2
    lim = 1 << (K-1)
    dp = [0] * lim
    dp[0] = 1
    for i, ch in enumerate(S):
        ndp = [0] * lim
        for mask in range(lim):
            cnt = dp[mask]
            if cnt == 0:
                continue
            # try A (0) or B (1)
            for c_bit in (0, 1):
                if ch == 'A' and c_bit == 1: continue
                if ch == 'B' and c_bit == 0: continue
                # if we form a window of length K ending here, check palindrome
                if i + 1 >= K:
                    pal = True
                    # check for t=0..K-1 chars at pos i-K+1+t
                    # for t in [0..K-2], char from mask at j_old = K-2 - t
                    # for t=K-1, char=c_bit
                    for t in range(K // 2):
                        # char at t
                        if t < K-1:
                            a = (mask >> (K-2 - t)) & 1
                        else:
                            a = c_bit
                        t2 = K-1 - t
                        if t2 < K-1:
                            b = (mask >> (K-2 - t2)) & 1
                        else:
                            b = c_bit
                        if a != b:
                            pal = False
                            break
                    if pal:
                        continue
                # update mask: new_mask bit0 = c_bit, bit j = old bit j-1
                new_mask = ((mask << 1) | c_bit) & (lim - 1)
                ndp[new_mask] = (ndp[new_mask] + cnt) % M
        dp = ndp
    ans = sum(dp) % M
    print(ans)

if __name__ == "__main__":
    main()