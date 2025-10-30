# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    import heapq

    def dijkstra(graph, start, n):
        dist = [-float('inf')] * (n + 1)
        dist[start] = 0
        pq = [(-0, start)]
        while pq:
            current_dist, u = heapq.heappop(pq)
            current_dist = -current_dist
            if current_dist < dist[u]:
                continue
            for v, weight in graph[u]:
                if dist[u] + weight > dist[v]:
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, (-dist[v], v))
        return dist

    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    index += 2

    graph = [[] for _ in range(N + 1)]

    for _ in range(M):
        l = int(input[index])
        d = int(input[index + 1])
        k = int(input[index + 2])
        c = int(input[index + 3])
        A = int(input[index + 4])
        B = int(input[index + 5])
        index += 6

        for i in range(k):
            t = l + i * d
            graph[A].append((B, t + c))

    dist = dijkstra(graph, 1, N)

    for i in range(2, N + 1):
        if dist[i] == -float('inf'):
            print("Unreachable")
        else:
            print(dist[i])

if __name__ == "__main__":
    main()