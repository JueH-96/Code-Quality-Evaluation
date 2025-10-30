import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N = int(input())
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # vertex 1 is already a leaf
    if len(adj[1]) <= 1:
        print(1)
        return

    visited = [False] * (N + 1)
    visited[1] = True                     # never enter vertex 1 again
    max_subtree = 0                       # size of the largest component after removing vertex 1

    for start in adj[1]:                  # one component per neighbour of 1
        if visited[start]:
            continue
        size = 0
        stack = [start]
        visited[start] = True
        while stack:                      # iterative DFS
            v = stack.pop()
            size += 1
            for nxt in adj[v]:
                if nxt == 1 or visited[nxt]:
                    continue
                visited[nxt] = True
                stack.append(nxt)
        max_subtree = max(max_subtree, size)

    # minimal number of operations = vertices we really have to delete
    #  = all vertices except the largest kept component + vertex 1 itself
    ans = N - max_subtree
    print(ans)

if __name__ == "__main__":
    main()