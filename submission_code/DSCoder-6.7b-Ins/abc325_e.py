import sys
import heapq

def read_input():
    N, A, B, C = map(int, sys.stdin.readline().split())
    D = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    return N, A, B, C, D

def solve():
    N, A, B, C, D = read_input()
    INF = float('inf')
    dist = [INF] * N
    dist[0] = 0
    queue = [(0, 0)]
    while queue:
        d, node = heapq.heappop(queue)
        if dist[node] < d:
            continue
        for next_node, D_ij in enumerate(D[node]):
            if D_ij == 0:
                continue
            alt = d + D_ij * A
            if alt < dist[next_node]:
                dist[next_node] = alt
                heapq.heappush(queue, (alt, next_node))
            alt = d + D_ij * B + C
            if alt < dist[next_node]:
                dist[next_node] = alt
                heapq.heappush(queue, (alt, next_node))
    return dist[-1]

print(solve())