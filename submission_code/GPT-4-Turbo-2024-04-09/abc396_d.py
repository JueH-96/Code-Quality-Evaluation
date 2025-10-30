import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    edges = []
    index = 2
    graph = {i: [] for i in range(1, N+1)}
    
    for _ in range(M):
        u = int(data[index])
        v = int(data[index+1])
        w = int(data[index+2])
        index += 3
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # Use a priority queue to implement a modified Dijkstra's algorithm for minimum XOR path
    # Priority queue stores tuples of (current_xor, current_vertex, visited_set)
    pq = []
    heapq.heappush(pq, (0, 1, {1}))
    
    while pq:
        current_xor, current_vertex, visited = heapq.heappop(pq)
        
        if current_vertex == N:
            print(current_xor)
            return
        
        for neighbor, weight in graph[current_vertex]:
            if neighbor not in visited:
                new_visited = set(visited)
                new_visited.add(neighbor)
                heapq.heappush(pq, (current_xor ^ weight, neighbor, new_visited))
    
if __name__ == "__main__":
    main()