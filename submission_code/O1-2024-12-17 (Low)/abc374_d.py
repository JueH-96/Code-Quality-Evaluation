def main():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = float(input_data[1])
    T = float(input_data[2])

    # Parse line segments
    segs = []
    idx = 3
    for _ in range(N):
        A, B, C, D = map(float, input_data[idx:idx+4])
        idx += 4
        segs.append(((A, B), (C, D)))

    # Precompute distances between endpoints of each segment
    # Also store each segment's length
    def dist(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    seg_length = []
    for ((x1, y1), (x2, y2)) in segs:
        seg_length.append(dist((x1, y1), (x2, y2)))

    # We'll do a DP with bitmasking:
    # dp[mask][i][k] = minimal time needed to have printed the segments in "mask"
    #   and to end at endpoint k (0 or 1) of segment i.
    # We will also have a "virtual start" state from (0,0) with mask=0.
    #
    # Transitions:
    #   From dp[mask][i][k] to dp[mask | (1<<j)][ j ][ 1-l ],
    #   cost = dp[mask][i][k] + (distance between current endpoint e_{i,k} and e_{j,l}) / S
    #          + (distance between e_{j,0} and e_{j,1}) / T
    #   where printing segment j means traveling from e_{j,l} to e_{j,1-l} with the laser on.
    #
    # For the initial step from "start state" (0,0) to printing any segment j,
    #   cost = dist((0,0), e_{j,l}) / S + seg_length[j] / T
    #   then we end at e_{j, 1-l}.

    INF = float('inf')
    # dp array: dimensions [1<<N][N][2]
    dp = [[[INF]*2 for _ in range(N)] for _ in range(1 << N)]

    # Initialize from (0,0) to first segment
    for j in range(N):
        for l in range(2):
            # l=0 means we start printing j from segs[j][0],
            # l=1 means from segs[j][1].
            start_pt = (0.0, 0.0)
            seg_start = segs[j][l]
            seg_end   = segs[j][1 - l]
            cost = dist(start_pt, seg_start) / S + seg_length[j] / T
            dp[1 << j][j][1 - l] = min(dp[1 << j][j][1 - l], cost)

    # Populate DP
    for mask in range(1 << N):
        for i in range(N):
            for k in range(2):
                if dp[mask][i][k] == INF:
                    continue
                base_time = dp[mask][i][k]
                # Current position = segs[i][k]
                curr_pt = segs[i][k]
                # Next segments to print
                for j in range(N):
                    if (mask & (1 << j)) != 0:  # already printed
                        continue
                    # Two ways to print segment j
                    for l in range(2):
                        seg_start = segs[j][l]
                        seg_end   = segs[j][1 - l]
                        cost = base_time \
                             + dist(curr_pt, seg_start) / S \
                             + seg_length[j] / T
                        if cost < dp[mask | (1 << j)][j][1 - l]:
                            dp[mask | (1 << j)][j][1 - l] = cost

    # Get result
    full_mask = (1 << N) - 1
    ans = INF
    for i in range(N):
        for k in range(2):
            ans = min(ans, dp[full_mask][i][k])

    # Print answer with appropriate precision
    print(ans)


# Call main to ensure execution
if __name__ == "__main__":
    main()