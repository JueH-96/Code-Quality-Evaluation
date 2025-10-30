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
    # We need to find the shortest path from 1 to any node that can reach 1
    # So, we perform BFS from 1 to find distances, and then for each node that can reach 1, we check the sum of distances
    
    # First, perform BFS from 1 to get distances
    dist_from_1 = [float('inf')] * (N+1)
    dist_from_1[1] = 0
    q = deque()
    q.append(1)
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist_from_1[v] == float('inf'):
                dist_from_1[v] = dist_from_1[u] + 1
                q.append(v)
    
    # Now, for each node that can reach 1, we need to find the shortest path from that node to 1
    # To do this, we reverse the graph and perform BFS from 1
    reverse_adj = [[] for _ in range(N+1)]
    for u in range(1, N+1):
        for v in adj[u]:
            reverse_adj[v].append(u)
    
    dist_to_1 = [float('inf')] * (N+1)
    dist_to_1[1] = 0
    q = deque()
    q.append(1)
    
    while q:
        u = q.popleft()
        for v in reverse_adj[u]:
            if dist_to_1[v] == float('inf'):
                dist_to_1[v] = dist_to_1[u] + 1
                q.append(v)
    
    # Now, for each node u, if dist_from_1[u] is finite and dist_to_1[u] is finite, then the cycle length is dist_from_1[u] + dist_to_1[u]
    min_cycle = float('inf')
    for u in range(1, N+1):
        if dist_from_1[u] != float('inf') and dist_to_1[u] != float('inf'):
            cycle_length = dist_from_1[u] + dist_to_1[u]
            if cycle_length < min_cycle:
                min_cycle = cycle_length
    
    if min_cycle != float('inf'):
        print(min_cycle)
    else:
        print(-1)

if __name__ == "__main__":
    main()