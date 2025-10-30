import heapq

def min_time_to_travel():
    import sys
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

    D = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(int(data[index]))
            index += 1
        D.append(row)

    # Dijkstra's algorithm
    def dijkstra():
        # (time, current_city, using_car)
        pq = [(0, 0, True), (0, 0, False)]
        dist = [[float('inf'), float('inf')] for _ in range(N)]
        dist[0][0] = 0
        dist[0][1] = 0

        while pq:
            time, city, using_car = heapq.heappop(pq)

            if city == N - 1:
                return min(dist[N - 1])

            for next_city in range(N):
                if D[city][next_city] == 0:
                    continue

                if using_car:
                    new_time = time + D[city][next_city] * A
                    if new_time < dist[next_city][0]:
                        dist[next_city][0] = new_time
                        heapq.heappush(pq, (new_time, next_city, True))
                    new_time = time + D[city][next_city] * B + C
                    if new_time < dist[next_city][1]:
                        dist[next_city][1] = new_time
                        heapq.heappush(pq, (new_time, next_city, False))
                else:
                    new_time = time + D[city][next_city] * B + C
                    if new_time < dist[next_city][1]:
                        dist[next_city][1] = new_time
                        heapq.heappush(pq, (new_time, next_city, False))

        return min(dist[N - 1])

    print(dijkstra())

min_time_to_travel()