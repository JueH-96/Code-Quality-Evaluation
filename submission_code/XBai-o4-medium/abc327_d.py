import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    adj = [[] for _ in range(N + 1)]
    for i in range(M):
        a = A[i]
        b = B[i]
        adj[a].append(b)
        adj[b].append(a)
    
    color = [-1] * (N + 1)
    
    for u in range(1, N + 1):
        if color[u] == -1:
            queue = deque()
            queue.append(u)
            color[u] = 0
            while queue:
                current = queue.popleft()
                for v in adj[current]:
                    if color[v] == -1:
                        color[v] = 1 - color[current]
                        queue.append(v)
                    elif color[v] == color[current]:
                        print("No")
                        return
    print("Yes")

if __name__ == "__main__":
    main()