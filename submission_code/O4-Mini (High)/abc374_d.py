import sys
import threading

def main():
    import sys, math

    data = sys.stdin.read().strip().split()
    it = iter(data)
    N = int(next(it))
    S = float(next(it))
    T = float(next(it))

    # Read segments
    # endpoints[i][0] = (A_i, B_i), endpoints[i][1] = (C_i, D_i)
    endpoints = []
    for _ in range(N):
        a = float(next(it)); b = float(next(it))
        c = float(next(it)); d = float(next(it))
        endpoints.append([(a,b),(c,d)])

    # Precompute emit times: time to print each segment
    emit_time = [0.0]*N
    for i in range(N):
        (x1,y1),(x2,y2) = endpoints[i]
        length = math.hypot(x2-x1, y2-y1)
        emit_time[i] = length / T

    # Precompute origin move times: time to go from (0,0) to start point of segment i with orientation dir
    # dir = 0: start at endpoints[i][0], dir=1: start at endpoints[i][1]
    origin_move = [[0.0]*2 for _ in range(N)]
    for i in range(N):
        for dir in (0,1):
            (sx,sy) = endpoints[i][dir]
            origin_move[i][dir] = math.hypot(sx, sy) / S

    # Precompute move time between any end of i (dir_i) to start of j (dir_j)
    move_time = [[ [ [0.0]*2 for _ in range(N) ] for __ in range(2) ] for ___ in range(N)]
    for i in range(N):
        for dir_i in (0,1):
            # end point of segment i if printed in dir_i:
            ex, ey = endpoints[i][1-dir_i]
            for j in range(N):
                for dir_j in (0,1):
                    sx, sy = endpoints[j][dir_j]
                    d = math.hypot(ex - sx, ey - sy)
                    move_time[i][dir_i][j][dir_j] = d / S

    INF = float('inf')
    # dp[mask][i][dir] = minimal time to have printed segments in 'mask', 
    # and last printed was segment i in orientation dir
    # mask size 2^N, i in [0..N-1], dir in [0,1]
    M = 1<<N
    dp = [ [ [INF]*2 for _ in range(N) ] for __ in range(M) ]

    # Initialize for one segment printed first
    for i in range(N):
        for dir in (0,1):
            m = 1<<i
            dp[m][i][dir] = origin_move[i][dir] + emit_time[i]

    # Fill DP
    for mask in range(M):
        # for each last segment in mask
        for last in range(N):
            if not (mask & (1<<last)): 
                continue
            for dir_last in (0,1):
                cur = dp[mask][last][dir_last]
                if cur == INF:
                    continue
                # try to print a next segment j
                rem = (~mask) & (M-1)
                # iterate over remaining segments
                j = rem
                # we can simply loop j from 0..N-1 and check
                for nxt in range(N):
                    if (mask >> nxt) & 1:
                        continue
                    for dir_nxt in (0,1):
                        nm = mask | (1<<nxt)
                        t = cur + move_time[last][dir_last][nxt][dir_nxt] + emit_time[nxt]
                        if t < dp[nm][nxt][dir_nxt]:
                            dp[nm][nxt][dir_nxt] = t

    # answer: all printed
    full = (1<<N) - 1
    ans = INF
    for i in range(N):
        for dir in (0,1):
            if dp[full][i][dir] < ans:
                ans = dp[full][i][dir]

    # print result
    # allowed error 1e-6
    # print with enough precision
    sys.stdout.write("{:.10f}
".format(ans))

if __name__ == "__main__":
    main()