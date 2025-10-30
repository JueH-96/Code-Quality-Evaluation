from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    adj = [[] for _ in range(N+1)]
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index+1])
        adj[a].append(b)
        index += 2
    
    # BFS to find the shortest cycle containing vertex 1
    visited = [False] * (N+1)
    distance = [0] * (N+1)
    queue = deque()
    queue.append(1)
    visited[1] = True
    distance[1] = 0
    
    min_cycle = float('inf')
    
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v == 1:
                # Found a cycle
                cycle_length = distance[u] + 1
                if cycle_length < min_cycle:
                    min_cycle = cycle_length
            if not visited[v]:
                visited[v] = True
                distance[v] = distance[u] + 1
                queue.append(v)
    
    if min_cycle != float('inf'):
        print(min_cycle)
    else:
        print(-1)

if __name__ == "__main__":
    main()