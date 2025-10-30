import sys
import numpy as np

def solve():
    import sys
    import numpy as np
    from sys import stdin
    import sys

    INF = 1 << 60

    input = sys.stdin.read().splitlines()
    idx = 0
    N, M, Q = map(int, input[idx].split())
    idx +=1

    roads = []
    A = []
    B = []
    C = []
    for _ in range(M):
        a, b, c = map(int, input[idx].split())
        A.append(a-1)
        B.append(b-1)
        C.append(c)
        roads.append( (a-1, b-1, c))
        idx +=1
    A = np.array(A, dtype=np.int32)
    B = np.array(B, dtype=np.int32)
    C = np.array(C, dtype=np.int64)

    queries = []
    road_remove_list = []
    for _ in range(Q):
        parts = input[idx].split()
        if parts[0] == '1':
            road_remove_list.append(int(parts[1])-1)
            queries.append( ('1', int(parts[1])-1) )
        else:
            x = int(parts[1])-1
            y = int(parts[2])-1
            queries.append( ('2', x, y) )
        idx +=1

    K = len(road_remove_list)
    road_remove_indices = []
    for i, q in enumerate(queries):
        if q[0] == '1':
            road_remove_indices.append(i)
    blocks = []
    prev =0
    for pos in road_remove_indices:
        blocks.append( (prev, pos) )
        prev = pos+1
    blocks.append( (prev, Q) )
    # Now, blocks is a list of (start, end) query indices
    # road_remove_list[i] will be removed after block i
    # Initialize roads_mask
    roads_mask = np.ones(M, dtype=bool)
    # Precompute roads data
    A_all = A
    B_all = B
    C_all = C
    # Prepare roads in Numpy arrays
    # Precompute sorted roads for faster operations
    for block_idx, (start, end) in enumerate(blocks):
        # Build active roads
        active = roads_mask
        A_active = A_all[active]
        B_active = B_all[active]
        C_active = C_all[active]
        # Initialize D
        D = np.full((N, N), INF, dtype=np.int64)
        D[A_active, B_active] = np.minimum(D[A_active, B_active], C_active)
        D[B_active, A_active] = np.minimum(D[B_active, A_active], C_active)
        np.fill_diagonal(D, 0)
        # Run Floyd-Warshall
        for k in range(N):
            D = np.minimum(D, D[:,k, np.newaxis] + D[k, :])
        # Now, D contains all-pairs shortest paths for this block
        # Answer the queries in this block
        block_queries = queries[start:end]
        output = []
        for q in block_queries:
            if q[0] == '2':
                x, y = q[1], q[2]
                dist = D[x, y]
                if dist < INF:
                    print(dist)
                else:
                    print(-1)
        # After the block, remove the next road if any
        if block_idx < K:
            road_to_remove = road_remove_list[block_idx]
            roads_mask[road_to_remove] = False