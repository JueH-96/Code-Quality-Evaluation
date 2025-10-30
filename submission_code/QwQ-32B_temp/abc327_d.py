import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Check if any pair has A_i == B_i
    for a, b in zip(A, B):
        if a == b:
            print("No")
            return
    
    # Build adjacency list
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    for a, b in zip(A, B):
        adj[a].append(b)
        adj[b].append(a)
    
    color = [-1] * (N + 1)
    
    for node in range(1, N + 1):
        if color[node] == -1:
            queue = deque()
            queue.append(node)
            color[node] = 0
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