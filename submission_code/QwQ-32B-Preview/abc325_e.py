import sys
import heapq

def dijkstra(graph, start, N):
    dist = [float('inf')] * N
    dist[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > dist[current_node]:
            continue
        
        for neighbor in range(N):
            weight = graph[current_node][neighbor]
            distance = current_distance + weight
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return dist

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    C = int(data[3])
    
    index = 4
    D = []
    for _ in range(N):
        row = list(map(int, data[index:index + N]))
        D.append(row)
        index += N
    
    car_distances = [[D[i][j] * A for j in range(N)] for i in range(N)]
    train_distances = [[D[i][j] * B + C for j in range(N)] for i in range(N)]
    
    car_dist = dijkstra(car_distances, 0, N)
    train_dist = dijkstra(train_distances, N-1, N)
    
    min_time = float('inf')
    for k in range(N):
        total_time = car_dist[k] + train_dist[k]
        if total_time < min_time:
            min_time = total_time
    
    print(min_time)

if __name__ == '__main__':
    main()