def main():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = float(input_data[1])
    T = float(input_data[2])

    coords = input_data[3:]
    segments = []
    idx = 0
    for _ in range(N):
        A = float(coords[idx]); B = float(coords[idx+1])
        C = float(coords[idx+2]); D = float(coords[idx+3])
        idx += 4
        segments.append(((A, B), (C, D)))

    # We'll label endpoints as follows:
    # For line i, endpoint 2*i is segments[i][0], endpoint 2*i+1 is segments[i][1].
    # We'll add endpoint 2*N as the initial origin (0,0).
    # So total of 2N+1 endpoints.
    E = []
    for i in range(N):
        E.append(segments[i][0])  # (A_i, B_i)
        E.append(segments[i][1])  # (C_i, D_i)
    E.append((0.0, 0.0))         # Start position at index 2N

    # Precompute distances between all endpoints
    # dist[i][j] = Euclidian distance between E[i] and E[j].
    # 0 <= i,j <= 2N
    def euclidian(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])
    size = 2*N + 1
    dist = [[0.0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            dist[i][j] = euclidian(E[i], E[j])

    # Precompute the length of each line segment (distance between its two endpoints)
    # line_length[i] = distance between E[2*i] and E[2*i+1]
    line_length = []
    for i in range(N):
        line_length.append(dist[2*i][2*i+1])

    # dp[mask][e]: the minimum time to have printed lines in mask, currently at endpoint e
    # mask is a subset of {0,1,...,N-1} indicating which line segments have been printed
    INF = float('inf')
    dp = [[INF]*(size) for _ in range(1<<N)]
    # Start with dp[0][2N] = 0, meaning no lines printed, at the origin index 2N
    dp[0][2*N] = 0.0

    # Iterate over all subsets of lines
    for mask in range(1<<N):
        for e in range(size):
            current_time = dp[mask][e]
            if current_time == INF:
                continue
            # Try to print a line segment i that's not yet in mask
            for i in range(N):
                if (mask & (1<<i)) == 0:
                    # We can print line i from E[2*i] to E[2*i+1], or from E[2*i+1] to E[2*i].
                    # Cost to move from current endpoint e to chosen start (either 2*i or 2*i+1) / S,
                    # plus line_length[i] / T to actually print it.
                    new_mask = mask | (1<<i)

                    # Direction: from E[2*i] -> E[2*i+1]
                    travel = dist[e][2*i] / S
                    draw = line_length[i] / T
                    new_e = 2*i+1
                    cand = current_time + travel + draw
                    if cand < dp[new_mask][new_e]:
                        dp[new_mask][new_e] = cand

                    # Direction: from E[2*i+1] -> E[2*i]
                    travel = dist[e][2*i+1] / S
                    draw = line_length[i] / T
                    new_e = 2*i
                    cand = current_time + travel + draw
                    if cand < dp[new_mask][new_e]:
                        dp[new_mask][new_e] = cand

    # We want dp[(1<<N)-1][e] minimized over e in [0..2N-1].
    # That will give the minimum time after printing all lines.
    ans = INF
    full_mask = (1<<N) - 1
    for e in range(size):
        if dp[full_mask][e] < ans:
            ans = dp[full_mask][e]

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()