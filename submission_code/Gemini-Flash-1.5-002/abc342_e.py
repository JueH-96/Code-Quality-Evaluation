# YOUR CODE HERE
import heapq

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        l, d, k, c, a, b = map(int, input().split())
        for i in range(k):
            t = l + i * d
            edges.append((a, b, t, t + c))

    results = [-float('inf')] * n
    results[n - 1] = float('inf')

    for start_node in range(n - 1):
        pq = []
        heapq.heappush(pq, (float('inf'), start_node))
        visited = set()

        while pq:
            arrival_time, current_node = heapq.heappop(pq)

            if (current_node, arrival_time) in visited:
                continue
            visited.add((current_node, arrival_time))

            
            max_arrival = -float('inf')
            for a, b, dep_time, arr_time in edges:
                if a == current_node and dep_time <= arrival_time:
                    max_arrival = max(max_arrival, arr_time)

            if max_arrival == -float('inf'):
                continue

            if current_node == n-1:
                results[start_node] = max(results[start_node], arrival_time)
                continue

            for a, b, dep_time, arr_time in edges:
                if a == current_node and dep_time <= arrival_time:
                    heapq.heappush(pq, (arr_time, b))

    for i in range(n - 1):
        if results[i] == -float('inf'):
            print("Unreachable")
        else:
            print(results[i])

solve()