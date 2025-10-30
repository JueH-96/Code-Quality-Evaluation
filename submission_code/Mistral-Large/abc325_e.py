import sys
import heapq

def dijkstra(n, graph, start):
    distances = [float('inf')] * n
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    A = int(data[index])
    index += 1
    B = int(data[index])
    index += 1
    C = int(data[index])
    index += 1

    D = [[0] * N for _ in range(N)]

    for i in range(N):
        for j in range(N):
            D[i][j] = int(data[index])
            index += 1

    car_graph = [[] for _ in range(N)]
    train_graph = [[] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if i != j:
                car_graph[i].append((j, D[i][j] * A))
                train_graph[i].append((j, D[i][j] * B + C))

    car_distances = dijkstra(N, car_graph, 0)
    train_distances = dijkstra(N, train_graph, N-1)

    min_time = float('inf')

    for i in range(N):
        min_time = min(min_time, car_distances[i] + train_distances[i])

    print(min_time)

if __name__ == "__main__":
    main()