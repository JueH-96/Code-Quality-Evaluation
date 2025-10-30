def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    # graph[a][b] = True means "a is stronger than b"
    graph = [[False]*N for _ in range(N)]

    for _ in range(M):
        A, B = map(int, input().split())
        graph[A-1][B-1] = True

    # Compute transitive closure using Floyd-Warshall-like approach
    for k in range(N):
        for i in range(N):
            for j in range(N):
                graph[i][j] = graph[i][j] or (graph[i][k] and graph[k][j])

    # A person i can be the strongest if there is no j such that j is stronger than i
    # i.e., no j with graph[j][i] == True
    possible_strongest = []
    for i in range(N):
        if all(not graph[j][i] for j in range(N) if j != i):
            possible_strongest.append(i + 1)

    # If exactly one possible strongest, print it; otherwise, print -1
    print(possible_strongest[0] if len(possible_strongest) == 1 else -1)

# Do not forget to call main()!
if __name__ == "__main__":
    main()