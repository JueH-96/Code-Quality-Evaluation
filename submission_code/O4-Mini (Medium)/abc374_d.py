import sys
import threading
import math

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    S = float(next(it))
    T = float(next(it))
    # Read segments
    pts = []  # pts[i] = [ (Ai,Bi), (Ci,Di) ]
    for _ in range(N):
        a = float(next(it)); b = float(next(it))
        c = float(next(it)); d = float(next(it))
        pts.append([(a, b), (c, d)])
    # Precompute segment lengths
    seg_len = []
    for i in range(N):
        (x1, y1), (x2, y2) = pts[i]
        seg_len.append(math.hypot(x2-x1, y2-y1))
    # Precompute distance from origin
    dist0 = [[0.0]*2 for _ in range(N)]
    for i in range(N):
        for d in (0,1):
            x,y = pts[i][d]
            dist0[i][d] = math.hypot(x, y)
    # Precompute distances between endpoints
    dist_e = [[ [[0.0]*2 for _ in range(N)] for __ in range(2)] for ___ in range(N)]
    for i in range(N):
        for d in (0,1):
            x1,y1 = pts[i][d]
            for j in range(N):
                for e in (0,1):
                    x2,y2 = pts[j][e]
                    dist_e[i][d][j][e] = math.hypot(x2-x1, y2-y1)
    INF = 1e18
    # dp[mask][i][d] = min time to print 'mask' finishing with segment i at endpoint d
    dp = [ [ [INF]*2 for _ in range(N) ] for __ in range(1<<N) ]
    # Initialize: from origin, pick any segment i, any start endpoint d_start
    for i in range(N):
        L = seg_len[i]
        em_time = L / T
        for d_start in (0,1):
            d_end = 1 - d_start
            move_time = dist0[i][d_start] / S
            m = 1 << i
            dp[m][i][d_end] = min(dp[m][i][d_end], move_time + em_time)
    # DP over masks
    FULL = (1<<N) - 1
    for mask in range(1<<N):
        if mask == 0: continue
        for i in range(N):
            if not (mask & (1<<i)): continue
            for d in (0,1):
                cur = dp[mask][i][d]
                if cur >= INF: continue
                # Try next segment j not in mask
                for j in range(N):
                    if mask & (1<<j): continue
                    L2 = seg_len[j]
                    em2 = L2 / T
                    for d_start in (0,1):
                        d_end = 1 - d_start
                        mv = dist_e[i][d][j][d_start] / S
                        nm = mask | (1<<j)
                        newt = cur + mv + em2
                        if newt < dp[nm][j][d_end]:
                            dp[nm][j][d_end] = newt
    # Get answer
    ans = INF
    for i in range(N):
        for d in (0,1):
            ans = min(ans, dp[FULL][i][d])
    # Print with sufficient precision
    print("{:.15f}".format(ans))

# Call main
main()