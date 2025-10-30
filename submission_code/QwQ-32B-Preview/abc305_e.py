from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    idx = 3
    for _ in range(M):
        a = int(data[idx]) - 1
        b = int(data[idx+1]) - 1
        adj[a].append(b)
        adj[b].append(a)
        idx += 2
    
    # Read guards' positions and staminas
    guards = []
    for _ in range(K):
        p = int(data[idx]) - 1
        h = int(data[idx+1])
        guards.append((p, h))
        idx += 2
    
    # Multi-source BFS
    from itertools import repeat
    visited = [False] * N
    guarded = set()
    queue = deque()
    
    for p, h in guards:
        queue.append((p, 0, h))
        visited[p] = True
    
    while queue:
        u, dist, h = queue.popleft()
        if dist <= h:
            guarded.add(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append((v, dist + 1, h))
    
    # Collect and sort guarded vertices
    guarded_list = sorted(guarded, key=lambda x: x + 1)
    print(len(guarded_list))
    print(' '.join(str(v + 1) for v in guarded_list))

if __name__ == '__main__':
    main()