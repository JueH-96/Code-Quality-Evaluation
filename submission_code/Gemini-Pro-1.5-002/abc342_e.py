# YOUR CODE HERE
import heapq

def solve():
    n, m = map(int, input().split())
    trains = []
    for _ in range(m):
        l, d, k, c, a, b = map(int, input().split())
        for i in range(k):
            trains.append((l + i * d, c, a - 1, b - 1))

    results = []
    for start_node in range(n - 1):
        max_arrival_time = -float('inf')
        
        pq = []
        heapq.heappush(pq, (-0, start_node))
        
        visited = set()

        while pq:
            current_time, current_node = heapq.heappop(pq)
            current_time *= -1

            if (current_time, current_node) in visited:
                continue
            visited.add((current_time, current_node))

            if current_node == n - 1:
                max_arrival_time = max(max_arrival_time, current_time)
                continue

            for departure_time, cost, from_node, to_node in trains:
                if from_node == current_node and departure_time >= current_time:
                    heapq.heappush(pq, (-(departure_time + cost), to_node))

        if max_arrival_time == -float('inf'):
            results.append("Unreachable")
        else:
            results.append(str(max_arrival_time))

    for result in results:
        print(result)

solve()