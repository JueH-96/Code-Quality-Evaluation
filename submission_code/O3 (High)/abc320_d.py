import sys
from collections import deque

def main() -> None:
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))

    adj = [[] for _ in range(N + 1)]  # 1-indexed

    for _ in range(M):
        a = int(next(it))
        b = int(next(it))
        x = int(next(it))
        y = int(next(it))
        adj[a].append((b, x, y))
        adj[b].append((a, -x, -y))  # reverse vector

    # coordinate arrays
    xs = [0] * (N + 1)
    ys = [0] * (N + 1)
    visited = [False] * (N + 1)

    # BFS/DFS from person 1
    q = deque([1])
    visited[1] = True
    xs[1] = 0
    ys[1] = 0

    while q:
        u = q.popleft()
        cu, du = xs[u], ys[u]
        for v, dx, dy in adj[u]:
            if not visited[v]:
                xs[v] = cu + dx
                ys[v] = du + dy
                visited[v] = True
                q.append(v)
            # (Optional) consistency check omitted – inputs guaranteed consistent

    out_lines = []
    for i in range(1, N + 1):
        if visited[i]:
            out_lines.append(f"{xs[i]} {ys[i]}")
        else:
            out_lines.append("undecidable")

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()