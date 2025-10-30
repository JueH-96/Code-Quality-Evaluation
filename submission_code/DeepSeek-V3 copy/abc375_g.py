import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    edges = []
    for _ in range(M):
        A = int(data[idx])
        B = int(data[idx+1])
        C = int(data[idx+2])
        edges.append((A-1, B-1, C))  # 0-based indexing
        idx += 3
    
    # Dijkstra's algorithm to find shortest distances from city 1 (0-based)
    def dijkstra(adj, start):
        dist = [float('inf')] * N
        dist[start] = 0
        heap = [(0, start)]
        while heap:
            current_dist, u = heapq.heappop(heap)
            if current_dist > dist[u]:
                continue
            for v, w in adj[u]:
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    heapq.heappush(heap, (dist[v], v))
        return dist
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for a, b, c in edges:
        adj[a].append((b, c))
        adj[b].append((a, c))
    
    # Compute shortest distance with all roads
    dist_all = dijkstra(adj, 0)
    shortest_all = dist_all[N-1]
    
    # For each road, compute shortest distance without it
    for i in range(M):
        a, b, c = edges[i]
        # Remove the road from the adjacency list
        adj[a].remove((b, c))
        adj[b].remove((a, c))
        # Compute shortest distance without this road
        dist_without = dijkstra(adj, 0)
        shortest_without = dist_without[N-1]
        # Restore the road
        adj[a].append((b, c))
        adj[b].append((a, c))
        # Determine if the shortest distances are different
        if shortest_all != shortest_without:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()