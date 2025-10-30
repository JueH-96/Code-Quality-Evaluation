# YOUR CODE HERE
import sys
from collections import defaultdict
from heapq import heappush, heappop

def main():
    N, M = map(int, input().split())
    trains = defaultdict(list)

    for _ in range(M):
        l, d, k, c, A, B = map(int, input().split())
        trains[B].append((A, l, d, k, c))

    def dijkstra(start, end):
        dist = [-float('inf')] * (N + 1)
        dist[end] = float('inf')
        pq = [(-float('inf'), end)]

        while pq:
            arrival_time, station = heappop(pq)
            arrival_time = -arrival_time

            if station == start:
                return arrival_time

            for prev, l, d, k, c in trains[station]:
                if arrival_time < l:
                    continue
                departures = min(k, (arrival_time - l) // d + 1)
                latest_departure = l + (departures - 1) * d
                if latest_departure > dist[prev]:
                    dist[prev] = latest_departure
                    heappush(pq, (-latest_departure, prev))

        return -float('inf')

    for i in range(1, N):
        result = dijkstra(i, N)
        print(result if result != -float('inf') else "Unreachable")

if __name__ == "__main__":
    main()