import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    adj = [[] for _ in range(N + 1)]
    for i in range(M):
        u = A[i]
        v = B[i]
        adj[u].append(v)
        adj[v].append(u)
    
    color = [-1] * (N + 1)
    for u in range(1, N + 1):
        if color[u] == -1:
            queue = deque()
            queue.append(u)
            color[u] = 0
            while queue:
                current = queue.popleft()
                for neighbor in adj[current]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[current]
                        queue.append(neighbor)
                    elif color[neighbor] == color[current]:
                        print("No")
                        return
    print("Yes")

if __name__ == "__main__":
    main()