import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Check if any pair has A_i == B_i
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
    possible = True
    
    for u in range(1, n + 1):
        if color[u] == -1:
            queue = deque([u])
            color[u] = 0
            while queue and possible:
                current = queue.popleft()
                for neighbor in adj[current]:
                    if color[neighbor] == -1:
                        color[neighbor] = color[current] ^ 1
                        queue.append(neighbor)
                    elif color[neighbor] == color[current]:
                        possible = False
                        break
                if not possible:
                    break
        if not possible:
            break
    
    print("Yes" if possible else "No")

if __name__ == "__main__":
    main()