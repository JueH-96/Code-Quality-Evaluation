import sys
import heapq

def dijkstra(N, A, B, C, D):
    # Initialize distances
    dist_car = [float('inf')] * N
    dist_train = [float('inf')] * N
    dist_car[0] = 0

    # Priority queue for Dijkstra's algorithm
    pq = [(0, 0, False)]  # (distance, city, is_train)

    while pq:
        curr_dist, curr_city, is_train = heapq.heappop(pq)

        # If we've already found a shorter path, skip this one
        if is_train and curr_dist > dist_train[curr_city]:
            continue
        if not is_train and curr_dist > dist_car[curr_city]:
            continue

        # Explore neighbors
        for next_city in range(N):
            if curr_city == next_city:
                continue

            next_dist_car = curr_dist + D[curr_city][next_city] * A
            next_dist_train = curr_dist + D[curr_city][next_city] * B + C

            # Update distances if we find a shorter path
            if next_dist_car < dist_car[next_city]:
                dist_car[next_city] = next_dist_car
                heapq.heappush(pq, (next_dist_car, next_city, False))
            if next_dist_train < dist_train[next_city]:
                dist_train[next_city] = next_dist_train
                heapq.heappush(pq, (next_dist_train, next_city, True))

    return min(dist_car[-1], dist_train[-1])

def main():
    N, A, B, C = map(int, input().split())
    D = [list(map(int, input().split())) for _ in range(N)]

    result = dijkstra(N, A, B, C, D)
    print(result)

if __name__ == "__main__":
    main()