import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    D = int(data[1])
    
    people = []
    index = 2
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        people.append((x, y))
        index += 2
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    D_sq = D * D
    for i in range(N):
        xi, yi = people[i]
        for j in range(i + 1, N):
            xj, yj = people[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            if dist_sq <= D_sq:
                adj[i].append(j)
                adj[j].append(i)
    
    # BFS to find all infected people
    visited = [False] * N
    q = deque()
    q.append(0)
    visited[0] = True
    
    while q:
        current = q.popleft()
        for neighbor in adj[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                q.append(neighbor)
    
    for i in range(N):
        print("Yes" if visited[i] else "No")

if __name__ == "__main__":
    main()