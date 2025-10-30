import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0]); M = int(line[1])
    # min_at_L[x] will store the minimal R among intervals with L = x
    # initialize to M+1 (i.e., "infinite")
    inf = M + 1
    min_at_L = [inf] * (M + 2)
    for _ in range(N):
        l, r = map(int, data.readline().split())
        if r < min_at_L[l]:
            min_at_L[l] = r

    # bestR[l] = minimum R among all intervals with L >= l
    # we do this in a rolling fashion
    best = inf
    ans = 0
    # iterate l from M down to 1
    for l in range(M, 0, -1):
        # incorporate intervals that start exactly at l
        if min_at_L[l] < best:
            best = min_at_L[l]
        # best now is min R among intervals with L >= l, or inf if none
        # valid r are those r < best, but also r >= l
        # count = max(0, min(M, best-1) - l + 1)
        # since best-1 <= M always when best<=M+1,
        # and if best == M+1, best-1 = M
        # so number of valid r is max(0, (best-1) - l + 1) = max(0, best - l)
        cnt = best - l
        if cnt > 0:
            ans += cnt

    # output the result
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()