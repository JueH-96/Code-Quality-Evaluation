# YOUR CODE HERE
import sys
import math
from heapq import heappush, heappop

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        current_dist, u = heappop(pq)
        if current_dist > dist[u]:
            continue
        for v, weight in graph[u]:
            distance = current_dist + weight
            if distance < dist[v]:
                dist[v] = distance
                heappush(pq, (distance, v))
    return dist

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    points = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(N)]
    
    graph = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = calculate_distance(points[i][0], points[i][1], points[j][0], points[j][1])
                graph[i].append((j, dist))
    
    min_penalty = [float('inf')] * N
    min_penalty[0] = 0
    
    for skipped in range(1, N-1):
        penalty = 2 ** (skipped - 1)
        for i in range(1, N-1):
            if min_penalty[i] == float('inf'):
                continue
            for j in range(i+1, N):
                if j == N-1:
                    min_penalty[j] = min(min_penalty[j], min_penalty[i] + calculate_distance(points[i][0], points[i][1], points[j][0], points[j][1]) + penalty)
                else:
                    min_penalty[j] = min(min_penalty[j], min_penalty[i] + calculate_distance(points[i][0], points[i][1], points[j][0], points[j][1]))
    
    print(min_penalty[N-1])

if __name__ == "__main__":
    main()