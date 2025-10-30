import sys
import itertools

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    bridges = []
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        t = int(input[ptr])
        ptr += 1
        bridges.append((u, v, t))
    
    # Initialize distance matrix with infinity
    INF = float('inf')
    dist = [[INF] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        dist[i][i] = 0
    
    # Populate initial minimal distances using the bridges
    for i in range(M):
        u, v, t = bridges[i]
        if t < dist[u][v]:
            dist[u][v] = t
            dist[v][u] = t
    
    # Floyd-Warshall algorithm to compute all-pairs shortest paths
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
        Bs = list(map(int, input[ptr:ptr+K]))
        ptr += K
        
        required_indices = [b - 1 for b in Bs]
        req = [(bridges[b][0], bridges[b][1], bridges[b][2]) for b in required_indices]
        sum_T = sum(t for (u, v, t) in req)
        
        min_total = INF
        
        # Iterate over all permutations of the required bridges' indices in req list
        for perm in itertools.permutations(range(K)):
            for mask in range(2 ** K):
                current = sum_T
                prev_node = 1  # Start from island 1
                for i in range(K):
                    idx = perm[i]
                    u, v, t = req[idx]
                    # Determine the start and end nodes based on the mask
                    if (mask >> i) & 1:
                        start, end = v, u
                    else:
                        start, end = u, v
                    current += dist[prev_node][start]
                    prev_node = end
                # Add the distance from the last node to island N
                current += dist[prev_node][N]
                if current < min_total:
                    min_total = current
        
        print(min_total)

if __name__ == "__main__":
    main()