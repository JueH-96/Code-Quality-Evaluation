import sys
import threading
def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    it = iter(data)
    N = int(next(it))
    S = float(next(it))
    T = float(next(it))
    # Read segments and their endpoints
    P = []  # P[i] = [(x0,y0),(x1,y1)]
    emit_cost = 0.0
    for _ in range(N):
        a = float(next(it)); b = float(next(it))
        c = float(next(it)); d = float(next(it))
        P.append([(a,b),(c,d)])
        # cost to emit that segment
        emit_cost += math.hypot(a-c, b-d) / T

    # Precompute cost to move (non-emitting) at speed S
    # cost_start[i][oi]: from (0,0) to P[i][oi]
    cost_start = [[0.0]*2 for _ in range(N)]
    for i in range(N):
        for oi in (0,1):
            x,y = P[i][oi]
            cost_start[i][oi] = math.hypot(x, y) / S

    # cost_between[i][oi][j][oj]: from end of printing i in orientation oi
    # to start endpoint P[j][oj]
    cost_between = [[ [ [0.0]*2 for _ in range(N) ] for __ in range(2) ] for ___ in range(N)]
    for i in range(N):
        for oi in (0,1):
            # end point of segment i when printed in orientation oi
            end_idx = 1 - oi
            ex, ey = P[i][end_idx]
            for j in range(N):
                if j == i: continue
                for oj in (0,1):
                    sx, sy = P[j][oj]
                    cost_between[i][oi][j][oj] = math.hypot(ex - sx, ey - sy) / S

    # DP over subsets
    INF = float('inf')
    full = 1<<N
    # dp[mask][i][oi]
    dp = [[[INF]*2 for _ in range(N)] for __ in range(full)]
    # initialize singletons
    for i in range(N):
        for oi in (0,1):
            dp[1<<i][i][oi] = cost_start[i][oi]

    # iterate masks
    for mask in range(full):
        # count bits optional
        for i in range(N):
            if not (mask & (1<<i)): continue
            for oi in (0,1):
                cur = dp[mask][i][oi]
                if cur == INF: continue
                # try extend to j
                for j in range(N):
                    if mask & (1<<j): continue
                    for oj in (0,1):
                        nm = mask | (1<<j)
                        c = cur + cost_between[i][oi][j][oj]
                        if c < dp[nm][j][oj]:
                            dp[nm][j][oj] = c

    # find best over full mask
    best = INF
    final_mask = full - 1
    if N == 0:
        best = 0.0
    else:
        for i in range(N):
            for oi in (0,1):
                if dp[final_mask][i][oi] < best:
                    best = dp[final_mask][i][oi]

    ans = best + emit_cost
    # print with sufficient precision
    print("{:.15f}".format(ans))

if __name__ == "__main__":
    main()