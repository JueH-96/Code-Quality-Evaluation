import sys
import itertools

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    INF = float('inf')
    dist = [[INF] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        dist[i][i] = 0

    bridges = [None] * (M + 1)
    adj = [[] for _ in range(N + 1)]

    for bridge_id in range(1, M + 1):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        t = int(input[ptr])
        ptr += 1
        bridges[bridge_id] = (u, v, t)
        adj[u].append((v, t))
        adj[v].append((u, t))
        if dist[u][v] > t:
            dist[u][v] = t
        if dist[v][u] > t:
            dist[v][u] = t

    # Floyd-Warshall algorithm
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            if dist[i][k] == INF:
                continue
            for j in range(1, N + 1):
                if dist[k][j] == INF:
                    continue
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    Q = int(input[ptr])
    ptr += 1

    results = []
    for _ in range(Q):
        K = int(input[ptr])
        ptr += 1
        B_list = list(map(int, input[ptr:ptr + K]))
        ptr += K

        required_bridges = [bridges[b] for b in B_list]
        min_time = INF

        # Iterate through all permutations of required bridges
        for perm in itertools.permutations(required_bridges):
            # Iterate through all possible direction masks
            for mask in range(0, 1 << K):
                current = 1
                total = 0
                valid = True
                for pos in range(K):
                    br = perm[pos]
                    u, v, t = br
                    direction = (mask >> pos) & 1
                    if direction == 0:
                        entry = u
                        exit_node = v
                    else:
                        entry = v
                        exit_node = u
                    # Add path from current to entry
                    step = dist[current][entry]
                    if step == INF:
                        valid = False
                        break
                    total += step + t
                    current = exit_node
                if not valid:
                    continue
                # Add path to N
                final_step = dist[current][N]
                if final_step == INF:
                    continue
                total += final_step
                if total < min_time:
                    min_time = total
        results.append(min_time)

    for res in results:
        print(res)

if __name__ == "__main__":
    main()