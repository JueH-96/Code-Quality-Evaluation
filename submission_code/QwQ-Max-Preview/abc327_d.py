import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))
    
    # Check if any pair has A_i == B_i
    for i in range(m):
        if a[i] == b[i]:
            print("No")
            return
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]  # 1-based indexing
    for i in range(m):
        u = a[i]
        v = b[i]
        adj[u].append(v)
        adj[v].append(u)
    
    color = [-1] * (n + 1)  # 1-based indexing
    
    for u in range(1, n + 1):
        if color[u] == -1:
            queue = deque([u])
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