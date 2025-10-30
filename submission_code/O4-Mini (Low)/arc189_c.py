import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, X = map(int, input().split())
    A = [0] + list(map(int, input().split()))
    B = [0] + list(map(int, input().split()))
    P = [0] + list(map(int, input().split()))
    Q = [0] + list(map(int, input().split()))

    # Build reverse adjacency lists for P and Q
    revP = [[] for _ in range(N+1)]
    revQ = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        revP[P[i]].append(i)
        revQ[Q[i]].append(i)

    from collections import deque

    # BFS on red‐graph to get dr[i] = distance (#ops) to move a red ball from i to X
    INF = 10**18
    dr = [INF]*(N+1)
    dq = deque([X])
    dr[X] = 0
    while dq:
        u = dq.popleft()
        for v in revP[u]:
            if dr[v] == INF:
                dr[v] = dr[u] + 1
                dq.append(v)

    # BFS on blue‐graph to get db[i]
    db = [INF]*(N+1)
    dq = deque([X])
    db[X] = 0
    while dq:
        u = dq.popleft()
        for v in revQ[u]:
            if db[v] == INF:
                db[v] = db[u] + 1
                dq.append(v)

    # If any initial ball cannot reach X, it's impossible
    for i in range(1, N+1):
        if A[i] == 1 and dr[i] == INF:
            print(-1)
            return
        if B[i] == 1 and db[i] == INF:
            print(-1)
            return

    visited = [False]*(N+1)
    visited[X] = True  # we don't count X itself

    # For each red ball at i, walk along P until X, marking nodes
    for i in range(1, N+1):
        if A[i] == 0:
            continue
        u = i
        # follow P-edges until we reach X; distances strictly decrease
        while not visited[u]:
            visited[u] = True
            u = P[u]
            # once u == X we'll stop marking further

    # For each blue ball at i, walk along Q until X
    for i in range(1, N+1):
        if B[i] == 0:
            continue
        u = i
        while not visited[u]:
            visited[u] = True
            u = Q[u]

    # Count how many marked boxes besides X
    answer = sum(1 for i in range(1, N+1) if visited[i] and i != X)
    print(answer)

if __name__ == "__main__":
    main()