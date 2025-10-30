import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    K = int(data[idx+2])
    idx += 3
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(data[idx])
        b = int(data[idx+1])
        adj[a].append(b)
        adj[b].append(a)
        idx += 2
    
    # Read guards
    guards = []
    for _ in range(K):
        p = int(data[idx])
        h = int(data[idx+1])
        guards.append((p, h))
        idx += 2
    
    # Initialize distance array
    distance = [float('inf')] * (N+1)
    
    # BFS for each guard
    for p, h in guards:
        q = deque()
        q.append((p, 0))
        distance[p] = min(distance[p], 0)
        while q:
            node, dist = q.popleft()
            if dist >= h:
                continue
            for neighbor in adj[node]:
                if distance[neighbor] > dist + 1:
                    distance[neighbor] = dist + 1
                    q.append((neighbor, dist + 1))
    
    # Collect guarded vertices
    guarded = []
    for v in range(1, N+1):
        if distance[v] != float('inf'):
            guarded.append(v)
    
    # Output
    print(len(guarded))
    print(' '.join(map(str, sorted(guarded))))

if __name__ == "__main__":
    main()