def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # For slightly faster I/O in Python (as Q can be large).
    # We'll parse tokens manually.

    # Read N, M, Q
    ptr = 0
    N = int(input_data[ptr]); ptr+=1
    M = int(input_data[ptr]); ptr+=1
    Q = int(input_data[ptr]); ptr+=1

    # Read roads (1-based -> 0-based)
    roads = []
    for _ in range(M):
        a = int(input_data[ptr]) - 1; ptr+=1
        b = int(input_data[ptr]) - 1; ptr+=1
        c = int(input_data[ptr]);     ptr+=1
        roads.append((a,b,c))

    # closure_time[i] will hold the query index (0-based) at which road i is closed.
    # If a road never closes, we'll keep it at Q (meaning effectively after all queries).
    closure_time = [Q]*M

    # Store all queries so we can answer distance queries afterward
    queries = []
    for i in range(Q):
        t = input_data[ptr]; ptr+=1
        if t == '1':
            # Road closure
            r_i = int(input_data[ptr]) - 1  # 1-based -> 0-based for road index
            ptr+=1
            closure_time[r_i] = i  # closes exactly at query i
            queries.append((1, r_i, None))
        else:
            # Distance query
            x = int(input_data[ptr]) - 1; ptr+=1
            y = int(input_data[ptr]) - 1; ptr+=1
            queries.append((2, x, y))

    # Collect all closures as (timeIndex, roadIndex), then sort by timeIndex
    closures = []
    for r_i in range(M):
        # if closure_time[r_i] < Q means it actually closes at some query
        # if = Q means never closed (we treat it as "time=Q" so it won't appear earlier)
        if closure_time[r_i] < Q:
            closures.append((closure_time[r_i], r_i))
    closures.sort(key=lambda x: x[0])  # sort by time

    # We'll define the order of closures: the s-th closure in sorted order is closures[s-1].
    # closure_index[i] = which number closure is road i in that sorted list, or -1 if it never closes
    closure_index = [-1]*M
    for idx, (tq, r_i) in enumerate(closures):
        closure_index[r_i] = idx

    # k = total number of closed roads
    k = len(closures)

    INF = 10**15

    # We will create adjacency matrices for states s=0..k.
    # adjacency[s] after Floyd-Warshall will be the matrix of shortest distances
    # with exactly s roads closed (in the order of closures[]).

    # Step 1: adjacency[0] with all roads open
    # Initialize a 2D list of size N*N
    adjacency = []
    dist0 = [[INF]*N for _ in range(N)]
    for i in range(N):
        dist0[i][i] = 0
    # Fill with direct distances from roads
    for i_road in range(M):
        (a,b,c) = roads[i_road]
        if c < dist0[a][b]:
            dist0[a][b] = c
            dist0[b][a] = c

    adjacency.append(dist0)

    # Function to copy matrix
    def copy_matrix(mat):
        return [row[:] for row in mat]

    # Floyd-Warshall in-place
    def floyd(mat):
        for kk in range(N):
            rowK = mat[kk]
            for ii in range(N):
                ik = mat[ii][kk]
                if ik == INF:
                    continue
                rowI = mat[ii]
                sK = 0
                for jj in range(N):
                    kj = rowK[jj]
                    nd = ik + kj
                    if nd < rowI[jj]:
                        rowI[jj] = nd

    # Compute Floyd-Warshall for state 0
    floyd(adjacency[0])

    # Build state s=1..k by copying state s-1 and removing one road, then FW again
    for s in range(1, k+1):
        # copy from adjacency[s-1]
        prev_mat = adjacency[s-1]
        cur_mat = copy_matrix(prev_mat)

        # remove the road that closes in the s-th closure
        # s-th closure in 1-based is closures[s-1] in 0-based
        _, road_id = closures[s-1]
        (aa,bb,cc) = roads[road_id]
        # set that edge cost to INF
        cur_mat[aa][bb] = INF
        cur_mat[bb][aa] = INF

        # Floyd-Warshall again
        floyd(cur_mat)
        adjacency.append(cur_mat)

    # Precompute for fast "get_state(timeIndex)" using binary search on closures' times
    from bisect import bisect_right
    closure_times_sorted = [c[0] for c in closures]
    def get_state(t):
        # number of closures with time <= t
        return bisect_right(closure_times_sorted, t)

    # Now process queries in order
    # For type=2 queries, we print the distance from x to y in the appropriate state
    out = []
    for i, q in enumerate(queries):
        if q[0] == 2:
            # distance query
            _, x, y = q
            s = get_state(i)
            dval = adjacency[s][x][y]
            if dval >= INF:
                out.append("-1")
            else:
                out.append(str(dval))

    print("
".join(out))