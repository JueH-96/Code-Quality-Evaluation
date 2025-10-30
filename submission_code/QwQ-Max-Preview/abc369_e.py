import sys
import itertools

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1

    bridges = []
    INF = float('inf')
    # Initialize distance matrix
    distance = [ [INF]*(N+1) for _ in range(N+1) ]
    for i in range(1, N+1):
        distance[i][i] = 0

    for _ in range(M):
        U = int(input[ptr])
        ptr +=1
        V = int(input[ptr])
        ptr +=1
        T = int(input[ptr])
        ptr +=1
        bridges.append( (U, V, T) )
        # Update the distance matrix for both directions
        if T < distance[U][V]:
            distance[U][V] = T
            distance[V][U] = T

    # Floyd-Warshall algorithm to compute all pairs shortest paths
    for k in range(1, N+1):
        for i in range(1, N+1):
            if distance[i][k] == INF:
                continue
            for j in range(1, N+1):
                if distance[k][j] == INF:
                    continue
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]

    Q = int(input[ptr])
    ptr +=1
    for _ in range(Q):
        K_i = int(input[ptr])
        ptr +=1
        B = list(map(int, input[ptr:ptr+K_i]))
        ptr += K_i
        # Collect the required bridges (convert to 0-based index)
        required_bridges = [ bridges[i-1] for i in B ]
        sum_T = sum( t for (u, v, t) in required_bridges )
        min_total = INF

        # Generate all permutations of the required bridges
        for perm in itertools.permutations(required_bridges):
            K = K_i
            # Iterate over all possible direction masks (0 to 2^K -1)
            for mask in range(0, 1 << K):
                current_sum_d = 0
                prev_node = 1
                for i in range(K):
                    u, v, t = perm[i]
                    # Determine direction based on mask's i-th bit
                    if (mask >> i) & 1:
                        start = v
                        end = u
                    else:
                        start = u
                        end = v
                    # Add distance from prev_node to start
                    current_sum_d += distance[prev_node][start]
                    prev_node = end
                # Add distance from last end to N
                current_sum_d += distance[prev_node][N]
                total = sum_T + current_sum_d
                if total < min_total:
                    min_total = total

        print( min_total )

if __name__ == '__main__':
    main()