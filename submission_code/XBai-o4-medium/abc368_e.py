import sys
import bisect
from collections import defaultdict, deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    X1 = int(input[ptr]); ptr +=1

    trains = []
    end_at = defaultdict(list)  # key: city, value: list of (T, index)

    for i in range(M):
        A = int(input[ptr]); ptr +=1
        B = int(input[ptr]); ptr +=1
        S = int(input[ptr]); ptr +=1
        T = int(input[ptr]); ptr +=1
        trains.append( (A, B, S, T) )
        end_at[B].append( (T, i) )

    # Preprocess for each city, sort the end_at lists and create ts and indices lists
    city_ts = dict()  # key: city, value: list of Ts
    city_indices = dict()  # key: city, value: list of indices

    for c in end_at:
        # sort by T
        sorted_pairs = sorted(end_at[c], key=lambda x: x[0])
        ts = [t for t, idx in sorted_pairs]
        indices = [idx for t, idx in sorted_pairs]
        city_ts[c] = ts
        city_indices[c] = indices

    # Build adjacency list
    adj = defaultdict(list)
    in_degree = [0]*M

    for j in range(M):
        A_j, B_j, S_j, T_j = trains[j]
        c = A_j
        if c not in city_ts:
            # no trains end here, so no edges
            continue
        ts_list = city_ts[c]
        indices_list = city_indices[c]
        # find all T_i <= S_j
        pos = bisect.bisect_right(ts_list, S_j)
        # add edges from indices_list[0...pos-1] to j
        for k in range(pos):
            i = indices_list[k]
            weight = ts_list[k] - S_j
            adj[i].append( (j, weight) )
            in_degree[j] += 1

    # Now perform topological sort using Kahn's algorithm
    top_order = []
    q = deque()

    for i in range(M):
        if in_degree[i] == 0:
            q.append(i)

    while q:
        u = q.popleft()
        top_order.append(u)
        for v, w in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)

    # Compute longest paths
    dist = [ -float('inf') ] * M
    dist[0] = 0  # starting node is train 0 (1-based train 1)

    for u in top_order:
        if dist[u] == -float('inf'):
            # not reachable from node 0, skip
            continue
        for (v, w) in adj[u]:
            if dist[v] < dist[u] + w:
                dist[v] = dist[u] + w

    # Compute X for each train
    X = [0]*M
    for i in range(M):
        val = X1 + (dist[i] if dist[i] != -float('inf') else -float('inf'))
        X[i] = max(val, 0)

    # Output X_2 ... X_M (original 1-based)
    # which is X[1], X[2], ..., X[M-1] in 0-based
    print(' '.join(map(str, X[1:])) )

if __name__ == '__main__':
    main()