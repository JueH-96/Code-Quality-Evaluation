import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    # Initialize adjacency list and road entries
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    road_entries = [[] for _ in range(M + 1)]  # road_entries[road_index] = list of (node, index_in_adj)
    
    for i in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        road_index = i + 1  # roads are 1-based
        # Add to A's adjacency list
        adj[A].append([B, C, road_index, False])
        road_entries[road_index].append((A, len(adj[A]) - 1))
        # Add to B's adjacency list
        adj[B].append([A, C, road_index, False])
        road_entries[road_index].append((B, len(adj[B]) - 1))
    
    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        if query_type == '1':
            # Close road i
            i = int(input[ptr])
            ptr += 1
            for (u, idx) in road_entries[i]:
                adj[u][idx][3] = True
        else:
            # Compute shortest path from x to y
            x = int(input[ptr])
            ptr += 1
            y = int(input[ptr])
            ptr += 1
            
            INF = 1 << 60
            dist = [INF] * (N + 1)
            dist[x] = 0
            heap = []
            heapq.heappush(heap, (0, x))
            
            while heap:
                d, u = heapq.heappop(heap)
                if u == y:
                    break
                if d > dist[u]:
                    continue
                for edge in adj[u]:
                    v = edge[0]
                    cost = edge[1]
                    closed = edge[3]
                    if not closed and dist[v] > dist[u] + cost:
                        dist[v] = dist[u] + cost
                        heapq.heappush(heap, (dist[v], v))
            
            print(-1 if dist[y] == INF else dist[y])

if __name__ == "__main__":
    main()