import sys
from itertools import permutations

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    bridges = []
    for _ in range(M):
        U = int(input[ptr])
        ptr += 1
        V = int(input[ptr])
        ptr += 1
        T = int(input[ptr])
        ptr += 1
        bridges.append((U, V, T))
    
    # Initialize distance matrix
    INF = float('inf')
    dist = [[INF] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        dist[i][i] = 0
    for U, V, T in bridges:
        if dist[U][V] > T:
            dist[U][V] = T
            dist[V][U] = T
    
    # Floyd-Warshall algorithm
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        K = int(input[ptr])
        ptr += 1
        B_list = list(map(int, input[ptr:ptr + K]))
        ptr += K
        required_bridges = [bridges[b - 1] for b in B_list]
        min_time = INF
        # Generate all permutations of the required bridges
        seen_perms = set()
        for perm in permutations(required_bridges):
            if perm in seen_perms:
                continue
            seen_perms.add(perm)
            K_perm = len(perm)
            # Generate all direction masks
            for mask in range(0, 1 << K_perm):
                total_time = 0
                prev_v = 1  # starting node is 1
                valid = True
                for i in range(K_perm):
                    B = perm[i]
                    U, V, T = B
                    direction = (mask >> i) & 1
                    if direction:
                        u = U
                        new_v = V
                    else:
                        u = V
                        new_v = U
                    # Compute distance from prev_v to u
                    if dist[prev_v][u] == INF:
                        valid = False
                        break
                    total_time += dist[prev_v][u]
                    # Add bridge time
                    total_time += T
                    prev_v = new_v
                if not valid:
                    continue
                # Add distance from prev_v to N
                if dist[prev_v][N] == INF:
                    continue
                total_time += dist[prev_v][N]
                if total_time < min_time:
                    min_time = total_time
        print(min_time if min_time != INF else -1)

if __name__ == '__main__':
    main()