import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    m = int(data[idx])
    idx += 1
    
    A = list(map(int, data[idx:idx+n]))
    idx += n
    
    adj = [[] for _ in range(n + 1)]
    
    for _ in range(m):
        u = int(data[idx])
        idx += 1
        v = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        
        # Add edge u -> v with weight B + A_v (v's weight)
        adj[u].append((v, b + A[v-1]))
        # Add edge v -> u with weight B + A_u (u's weight)
        adj[v].append((u, b + A[u-1]))
    
    # Dijkstra's algorithm
    INF = float('inf')
    distance = [INF] * (n + 1)
    distance[1] = A[0]  # Initial distance for node 1
    
    heap = []
    heapq.heappush(heap, (distance[1], 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > distance[u]:
            continue
        for v, w in adj[u]:
            if distance[v] > distance[u] + w:
                distance[v] = distance[u] + w
                heapq.heappush(heap, (distance[v], v))
    
    # Prepare the output for nodes 2 to n
    output = []
    for i in range(2, n+1):
        output.append(str(distance[i]))
    print(' '.join(output))

if __name__ == '__main__':
    main()