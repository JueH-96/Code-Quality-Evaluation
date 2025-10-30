import sys
from collections import deque

def main():
    # Read input
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Check if any A[i] == B[i]
    for i in range(M):
        if A[i] == B[i]:
            print("No")
            return
    
    # Build the adjacency list
    adj = [[] for _ in range(N + 1)]
    for i in range(M):
        u = A[i]
        v = B[i]
        adj[u].append(v)
        adj[v].append(u)
    
    # Check bipartition using BFS
    color = [-1] * (N + 1)
    for i in range(1, N + 1):
        if color[i] == -1:
            queue = deque()
            queue.append(i)
            color[i] = 0
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        print("No")
                        return
    print("Yes")

if __name__ == "__main__":
    main()