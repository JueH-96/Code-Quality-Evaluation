def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    X_1 = int(data[idx])
    idx += 1

    trains = []
    for _ in range(M):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        S = int(data[idx])
        idx += 1
        T = int(data[idx])
        idx += 1
        trains.append((A, B, S, T))

    edges = []

    for i in range(M):
        A_i, B_i, S_i, T_i = trains[i]
        for j in range(M):
            if i == j:
                continue
            A_j, B_j, S_j, T_j = trains[j]
            if B_i == A_j and T_i <= S_j:
                c = T_i - S_j
                edges.append((j + 1, i + 1, -c))  # Edge from j+1 to i+1 with weight -c

    # Add constraints X_i >= 0 by adding edges from each node to the virtual node 0 with weight 0
    for i in range(1, M + 1):
        edges.append((i, 0, 0))

    # Initialize distances
    INF = float('inf')
    distance = [INF] * (M + 1)
    distance[0] = 0  # Virtual node 0 has distance 0
    distance[1] = X_1  # Set X_1 to the given value

    # Run Bellman-Ford algorithm
    for _ in range(M):
        updated = False
        for u, v, w in edges:
            if distance[u] != INF and distance[v] > distance[u] + w:
                distance[v] = distance[u] + w
                updated = True
        if not updated:
            break

    # Ensure non-negative values
    for i in range(1, M + 1):
        distance[i] = max(0, distance[i])

    # Prepare the output for X_2 to X_M
    output = []
    for i in range(2, M + 1):
        output.append(str(distance[i]))

    print(' '.join(output))

if __name__ == '__main__':
    main()