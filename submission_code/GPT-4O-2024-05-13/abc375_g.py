# YOUR CODE HERE
import heapq
import sys
input = sys.stdin.read

def dijkstra(n, graph, start):
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v, weight in graph[u]:
            distance = current_dist + weight
            if distance < dist[v]:
                dist[v] = distance
                heapq.heappush(pq, (distance, v))
    return dist

def main():
    data = input().strip().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx + 1])
    idx += 2
    
    roads = []
    graph = [[] for _ in range(N + 1)]
    
    for i in range(M):
        A = int(data[idx])
        B = int(data[idx + 1])
        C = int(data[idx + 2])
        idx += 3
        roads.append((A, B, C))
        graph[A].append((B, C))
        graph[B].append((A, C))
    
    # Compute the shortest path from 1 to N with all roads
    dist_all = dijkstra(N, graph, 1)
    shortest_all = dist_all[N]
    
    results = []
    
    for i in range(M):
        A, B, C = roads[i]
        
        # Remove the road (A, B, C)
        graph[A].remove((B, C))
        graph[B].remove((A, C))
        
        # Compute the shortest path from 1 to N without this road
        dist_without = dijkstra(N, graph, 1)
        shortest_without = dist_without[N]
        
        # Compare the shortest paths
        if shortest_all != shortest_without:
            results.append("Yes")
        else:
            results.append("No")
        
        # Add the road back
        graph[A].append((B, C))
        graph[B].append((A, C))
    
    # Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()