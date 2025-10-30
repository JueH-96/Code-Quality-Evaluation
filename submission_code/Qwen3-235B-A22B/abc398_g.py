import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1

    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(input[idx]); idx += 1
        v = int(input[idx]); idx += 1
        adj[u].append(v)
        adj[v].append(u)

    color = [-1] * (N + 1)
    a = 0
    b = 0

    for u in range(1, N + 1):
        if color[u] == -1:
            # BFS to determine bipartition
            queue = deque()
            queue.append(u)
            color[u] = 0
            s = 1  # count of color 0
            t = 0  # count of color 1
            while queue:
                v = queue.popleft()
                for nei in adj[v]:
                    if color[nei] == -1:
                        color[nei] = 1 - color[v]
                        if color[nei] == 0:
                            s += 1
                        else:
                            t += 1
                        queue.append(nei)
            a += s
            b += t

    K = a * b - M
    if K <= 0:
        print("Takahashi")
    else:
        print("Aoki" if K % 2 == 1 else "Takahashi")

if __name__ == '__main__':
    main()