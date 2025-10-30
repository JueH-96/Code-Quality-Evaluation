import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Check if any A_i == B_i
    for a, b in zip(A, B):
        if a == b:
            print("No")
            return
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for a, b in zip(A, B):
        adj[a].append(b)
        adj[b].append(a)
    
    color = [-1] * (n + 1)
    
    for u in range(1, n + 1):
        if color[u] == -1:
            queue = deque()
            queue.append(u)
            color[u] = 0
            while queue:
                current = queue.popleft()
                for v in adj[current]:
                    if color[v] == -1:
                        color[v] = color[current] ^ 1
                        queue.append(v)
                    elif color[v] == color[current]:
                        print("No")
                        return
    print("Yes")

if __name__ == "__main__":
    main()