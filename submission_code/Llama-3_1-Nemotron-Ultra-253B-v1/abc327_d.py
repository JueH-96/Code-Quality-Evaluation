import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Check for any A[i] == B[i]
    for a, b in zip(A, B):
        if a == b:
            print("No")
            return
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for a, b in zip(A, B):
        adj[a].append(b)
        adj[b].append(a)
    
    # Check bipartition using BFS
    color = [-1] * (n + 1)
    for u in range(1, n + 1):
        if color[u] == -1:
            queue = deque([u])
            color[u] = 0
            while queue:
                current = queue.popleft()
                for neighbor in adj[current]:
                    if color[neighbor] == -1:
                        color[neighbor] = color[current] ^ 1
                        queue.append(neighbor)
                    elif color[neighbor] == color[current]:
                        print("No")
                        return
    print("Yes")

if __name__ == "__main__":
    main()