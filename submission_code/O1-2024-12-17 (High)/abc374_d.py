import sys
import math

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = float(data[1])
    T = float(data[2])
    
    # Read segment endpoints
    segments = []
    idx = 3
    for _ in range(N):
        A = float(data[idx]); B = float(data[idx+1])
        C = float(data[idx+2]); D = float(data[idx+3])
        idx += 4
        segments.append(((A,B),(C,D)))
    
    # Precompute lengths of each segment
    # length[i] = distance from segments[i][0] to segments[i][1]
    def dist(p, q):
        return math.hypot(q[0] - p[0], q[1] - p[1])
    
    length = [dist(seg[0], seg[1]) for seg in segments]
    
    # costStart[i][0] = cost to go from (0,0) to segments[i][0] by speed S,
    #                   then print the segment from [0] to [1] by speed T
    # costStart[i][1] = cost to go from (0,0) to segments[i][1] by speed S,
    #                   then print the segment from [1] to [0] by speed T
    costStart = [[0.0, 0.0] for _ in range(N)]
    zero = (0.0, 0.0)
    for i in range(N):
        costStart[i][0] = dist(zero, segments[i][0]) / S + length[i] / T
        costStart[i][1] = dist(zero, segments[i][1]) / S + length[i] / T
    
    # costTrans[i][e][j][e'] = time to move from the end of segment i with orientation e
    #                          to the start of segment j with orientation e', plus
    #                          the time to print segment j in that orientation.
    #
    # Orientation e=0 means we printed segment i from [0] to [1], so we end at [1].
    # Orientation e=1 means we printed segment i from [1] to [0], so we end at [0].
    # Similarly for j: e'=0 => start at j[0] -> j[1], e'=1 => start at j[1] -> j[0].
    costTrans = [[[[0.0 for _ in range(2)] for _ in range(N)] for _ in range(2)] for _ in range(N)]
    
    for i in range(N):
        # Endpoints for segment i
        e0_i = segments[i][0]  # (A_i, B_i)
        e1_i = segments[i][1]  # (C_i, D_i)
        for j in range(N):
            if i == j:
                continue
            # Endpoints for segment j
            e0_j = segments[j][0]
            e1_j = segments[j][1]
            # If i is printed "forward" => ended at e1_i
            # j is to be printed "forward" => start at e0_j
            costTrans[i][0][j][0] = dist(e1_i, e0_j)/S + length[j]/T
            # i forward => ended at e1_i
            # j reverse => start at e1_j
            costTrans[i][0][j][1] = dist(e1_i, e1_j)/S + length[j]/T
            # i reverse => ended at e0_i
            # j forward => start at e0_j
            costTrans[i][1][j][0] = dist(e0_i, e0_j)/S + length[j]/T
            # i reverse => ended at e0_i
            # j reverse => start at e1_j
            costTrans[i][1][j][1] = dist(e0_i, e1_j)/S + length[j]/T
    
    # We will use DP[mask][i][e] = minimum time to have printed the segments in "mask",
    # ending with segment i in orientation e.
    INF = float('inf')
    DP = [[[INF]*2 for _ in range(N)] for _ in range(1<<N)]
    
    # Initialize base cases: printing each segment first, in either orientation
    for i in range(N):
        DP[1<<i][i][0] = costStart[i][0]
        DP[1<<i][i][1] = costStart[i][1]
    
    # Fill DP
    for mask in range(1<<N):
        for i in range(N):
            for e in range(2):
                base_cost = DP[mask][i][e]
                if base_cost == INF:
                    continue
                # Try printing a new segment j not in mask
                for j in range(N):
                    if (mask & (1<<j)) != 0:
                        continue
                    for e2 in range(2):
                        new_mask = mask | (1<<j)
                        cand = base_cost + costTrans[i][e][j][e2]
                        if cand < DP[new_mask][j][e2]:
                            DP[new_mask][j][e2] = cand
    
    # The answer is the minimum DP value when mask == (1<<N)-1
    full_mask = (1<<N) - 1
    ans = INF
    for i in range(N):
        for e in range(2):
            if DP[full_mask][i][e] < ans:
                ans = DP[full_mask][i][e]
    
    print(ans)

# Do not forget to call main()
main()