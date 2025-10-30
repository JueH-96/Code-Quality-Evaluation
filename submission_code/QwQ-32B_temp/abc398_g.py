import sys
from sys import stdin
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v = map(int, stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    visited = [False] * (N + 1)
    color = [-1] * (N + 1)
    total_a = 0
    total_b = 0

    for u in range(1, N + 1):
        if not visited[u]:
            q = deque()
            q.append(u)
            visited[u] = True
            color[u] = 0
            a = 1
            b = 0
            while q:
                v = q.popleft()
                for neighbor in adj[v]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        new_color = 1 - color[v]
                        color[neighbor] = new_color
                        if new_color == 0:
                            a += 1
                        else:
                            b += 1
                        q.append(neighbor)
            total_a += a
            total_b += b

    available = total_a * total_b - M
    if available % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

if __name__ == "__main__":
    main()