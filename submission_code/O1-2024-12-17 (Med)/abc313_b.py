def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = list(map(int, input_data[2:]))

    # adjacency[i][j] = True means "i is stronger than j"
    adjacency = [[False]*N for _ in range(N)]

    idx = 0
    for _ in range(M):
        A = edges[idx]; B = edges[idx+1]
        idx += 2
        adjacency[A-1][B-1] = True

    # Compute the transitive closure (Floyd-Warshall or similar)
    for k in range(N):
        for i in range(N):
            if adjacency[i][k]:
                for j in range(N):
                    adjacency[i][j] = adjacency[i][j] or adjacency[k][j]

    # Check how many can be deduced to be stronger than all others
    candidates = []
    for i in range(N):
        # Check if i is stronger than every other j
        if all(i == j or adjacency[i][j] for j in range(N)):
            candidates.append(i+1)

    # If exactly one such candidate, print it; otherwise -1
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

# Do not forget to call main()
if __name__ == "__main__":
    main()