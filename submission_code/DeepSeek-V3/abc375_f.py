import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    Q = int(data[idx+2])
    idx += 3
    
    # Read roads
    roads = []
    for _ in range(M):
        A = int(data[idx])
        B = int(data[idx+1])
        C = int(data[idx+2])
        roads.append((A, B, C))
        idx += 3
    
    # Initialize closed roads set
    closed = set()
    
    # Process queries
    for _ in range(Q):
        query_type = int(data[idx])
        if query_type == 1:
            i = int(data[idx+1]) - 1
            closed.add(i)
            idx += 2
        else:
            x = int(data[idx+1])
            y = int(data[idx+2])
            idx += 3
            
            # Build graph without closed roads
            adj = [[] for _ in range(N+1)]
            for j in range(M):
                if j not in closed:
                    A, B, C = roads[j]
                    adj[A].append((B, C))
                    adj[B].append((A, C))
            
            # Dijkstra's algorithm
            dist = [float('inf')] * (N+1)
            dist[x] = 0
            heap = []
            heapq.heappush(heap, (0, x))
            
            while heap:
                current_dist, u = heapq.heappop(heap)
                if u == y:
                    break
                if current_dist > dist[u]:
                    continue
                for v, w in adj[u]:
                    if dist[v] > dist[u] + w:
                        dist[v] = dist[u] + w
                        heapq.heappush(heap, (dist[v], v))
            
            if dist[y] != float('inf'):
                print(dist[y])
            else:
                print(-1)

if __name__ == "__main__":
    main()