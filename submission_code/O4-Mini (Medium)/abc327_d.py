import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Build graph, and immediately reject self-loops
    adj = [[] for _ in range(N)]
    for i in range(M):
        u = A[i] - 1
        v = B[i] - 1
        if u == v:
            print("No")
            return
        adj[u].append(v)
        adj[v].append(u)

    # -1 = unvisited, 0/1 = bipartite color
    color = [-1] * N
    for start in range(N):
        if color[start] == -1 and adj[start]:
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        queue.append(v)
                    elif color[v] == color[u]:
                        print("No")
                        return

    print("Yes")

if __name__ == "__main__":
    main()