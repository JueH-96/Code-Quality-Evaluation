import sys
from collections import deque

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:     # no input
        return
    it = iter(data)
    N = int(next(it))
    D = int(next(it))
    coords = [(int(next(it)), int(next(it))) for _ in range(N)]

    D2 = D * D               # compare squared distances to avoid sqrt
    adj = [[] for _ in range(N)]

    # build the graph: connect i and j if their distance ≤ D
    for i in range(N):
        xi, yi = coords[i]
        for j in range(i + 1, N):
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy <= D2:
                adj[i].append(j)
                adj[j].append(i)

    # BFS/DFS starting from person 0 (person 1 in 1-based indexing)
    infected = [False] * N
    q = deque([0])
    infected[0] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not infected[v]:
                infected[v] = True
                q.append(v)

    # output results
    out_lines = ["Yes" if infected[i] else "No" for i in range(N)]
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()