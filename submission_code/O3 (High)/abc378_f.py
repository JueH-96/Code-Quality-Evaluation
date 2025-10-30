import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N = int(input())
    adj = [[] for _ in range(N)]
    for _ in range(N - 1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)

    deg = [len(neigh) for neigh in adj]          # degree of every vertex
    visited = [False] * N                        # used while scanning degree-3 components

    ans = 0

    for v in range(N):
        if deg[v] != 3 or visited[v]:
            continue

        # explore one connected component consisting only of degree-3 vertices
        stack = [v]
        visited[v] = True
        deg2_vertices = set()                    # degree-2 vertices adjacent to this component

        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if deg[nxt] == 3:
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(nxt)
                elif deg[nxt] == 2:              # potential cycle end-point
                    deg2_vertices.add(nxt)

        t = len(deg2_vertices)                   # how many degree-2 neighbours
        ans += t * (t - 1) // 2                  # all unordered pairs of them

    print(ans)


if __name__ == "__main__":
    main()