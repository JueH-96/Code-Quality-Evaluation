# YOUR CODE HERE
import sys
import threading
from collections import deque

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)

    visited = [False] * (N + 1)
    depth = [0] * (N + 1)
    parent = [0] * (N + 1)
    min_cycle_length = -1

    from collections import deque
    queue = deque()
    start = 1
    visited[start] = True
    depth[start] = 0
    queue.append(start)

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                depth[v] = depth[u] + 1
                parent[v] = u
                queue.append(v)
            else:
                # Check for back edges
                if parent[u] != v:
                    cycle_len = depth[u] - depth[v] + 1
                    if min_cycle_length == -1 or cycle_len < min_cycle_length:
                        min_cycle_length = cycle_len

    print(min_cycle_length)