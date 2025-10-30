import sys
import heapq

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    graph = [[] for _ in range(N)]
    index = 1
    for i in range(1, N):
        A = int(data[index])
        B = int(data[index + 1])
        X = int(data[index + 2]) - 1  # Convert to 0-based indexing
        graph[i - 1].append((i, A))
        graph[i - 1].append((X, B))
        index += 3
    
    distances = [float('inf')] * N
    distances[0] = 0
    priority_queue = [(0, 0)]  # (distance, node)
    
    while priority_queue:
        dist, node = heapq.heappop(priority_queue)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(priority_queue, (new_dist, neighbor))
    
    print(distances[N - 1])

if __name__ == "__main__":
    main()