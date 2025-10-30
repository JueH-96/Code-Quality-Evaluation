# YOUR CODE HERE
import sys
import heapq

def dijkstra(N, A, B, C, D):
    # Initialize distances
    dist = [[float('inf')] * 2 for _ in range(N)]
    dist[0][0] = 0  # Starting point with car

    # Min-heap for Dijkstra's algorithm
    pq = [(0, 0, 0)]  # (current distance, city, mode: 0=car, 1=train)

    while pq:
        current_dist, u, mode = heapq.heappop(pq)

        if current_dist > dist[u][mode]:
            continue

        for v in range(N):
            if v == u:
                continue

            if mode == 0:  # If currently using car
                car_time = current_dist + D[u][v] * A
                if car_time < dist[v][0]:
                    dist[v][0] = car_time
                    heapq.heappush(pq, (car_time, v, 0))

                train_time = current_dist + D[u][v] * B + C
                if train_time < dist[v][1]:
                    dist[v][1] = train_time
                    heapq.heappush(pq, (train_time, v, 1))
            else:  # If currently using train
                train_time = current_dist + D[u][v] * B + C
                if train_time < dist[v][1]:
                    dist[v][1] = train_time
                    heapq.heappush(pq, (train_time, v, 1))

    return min(dist[N-1][0], dist[N-1][1])

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0

    N = int(input[index])
    A = int(input[index + 1])
    B = int(input[index + 2])
    C = int(input[index + 3])
    index += 4

    D = []
    for i in range(N):
        row = list(map(int, input[index:index + N]))
        D.append(row)
        index += N

    result = dijkstra(N, A, B, C, D)
    print(result)

if __name__ == "__main__":
    main()