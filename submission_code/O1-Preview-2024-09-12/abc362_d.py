# YOUR CODE HERE
import sys
import threading
def main():
    import heapq
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    graph = [[] for _ in range(N)]
    for _ in range(M):
        U_j, V_j, B_j = map(int, sys.stdin.readline().split())
        U_j -=1
        V_j -=1
        graph[U_j].append((V_j, B_j))
        graph[V_j].append((U_j, B_j))

    INF = float('inf')
    dist = [INF]*N
    dist[0] = A[0]
    hq = [(dist[0], 0)]
    while hq:
        curr_dist_u, u = heapq.heappop(hq)
        if dist[u] < curr_dist_u:
            continue
        for v, b in graph[u]:
            new_dist = dist[u] + b + A[v]
            if dist[v] > new_dist:
                dist[v] = new_dist
                heapq.heappush(hq, (dist[v], v))
    result = [str(int(dist[i])) for i in range(1, N)]
    print(' '.join(result))

threading.Thread(target=main).start()
# YOUR CODE ENDS HERE