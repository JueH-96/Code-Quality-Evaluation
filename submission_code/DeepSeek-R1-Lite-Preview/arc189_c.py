def main():
    import sys
    from collections import deque

    sys.setrecursionlimit(1 << 25)
    N, X = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))

    X -= 1  # 0-indexed

    # Function to find the cycle containing X in permutation perm
    def find_cycle(perm, X):
        visited = [False] * N
        cycle = []
        current = X
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = perm[current] - 1  # 0-indexed
        return cycle

    # Find cycles containing X in P and Q
    cycle_P = find_cycle(P, X)
    cycle_Q = find_cycle(Q, X)

    # Create adjacency lists for dependencies
    adj_P = [[] for _ in range(N)]
    adj_Q = [[] for _ in range(N)]
    for i in range(N):
        adj_P[P[i]-1].append(i)
        adj_Q[Q[i]-1].append(i)

    # BFS to find distances from X in P's cycle
    def bfs_P(cycle_P, adj_P, X):
        dist = [-1] * N
        queue = deque()
        queue.append(X)
        dist[X] = 0
        while queue:
            u = queue.popleft()
            for v in adj_P[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    # BFS to find distances from X in Q's cycle
    def bfs_Q(cycle_Q, adj_Q, X):
        dist = [-1] * N
        queue = deque()
        queue.append(X)
        dist[X] = 0
        while queue:
            u = queue.popleft()
            for v in adj_Q[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    dist_P = bfs_P(cycle_P, adj_P, X)
    dist_Q = bfs_Q(cycle_Q, adj_Q, X)

    # Find maximum distances for red and blue balls
    max_red = 0
    for i in cycle_P:
        if A[i]:
            if dist_P[i] == -1:
                print(-1)
                return
            max_red = max(max_red, dist_P[i])

    max_blue = 0
    for i in cycle_Q:
        if B[i]:
            if dist_Q[i] == -1:
                print(-1)
                return
            max_blue = max(max_blue, dist_Q[i])

    # The minimum number of operations is the maximum of max_red and max_blue
    print(max(max_red, max_blue))

if __name__ == "__main__":
    main()