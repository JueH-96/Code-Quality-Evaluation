import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    B = list(map(int, data[2+M:2+2*M]))
    
    # Check for any A_i == B_i
    if any(A[i] == B[i] for i in range(M)):
        print("No")
        return
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for a, b in zip(A, B):
        adj[a].append(b)
        adj[b].append(a)
    
    # Initialize color array
    color = [-1] * (N+1)
    
    def bfs(start):
        queue = deque([start])
        color[start] = 0
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
        return True
    
    is_possible = True
    for node in range(1, N+1):
        if color[node] == -1:
            if not bfs(node):
                is_possible = False
                break
    
    print("Yes" if is_possible else "No")

if __name__ == "__main__":
    main()