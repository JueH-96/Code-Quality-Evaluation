# YOUR CODE HERE
import sys
import threading
import heapq
def main():
    import sys
    import threading
    sys.setrecursionlimit(1 << 25)
    N, M, K = map(int, sys.stdin.readline().split())
    N = int(N)
    M = int(M)
    K = int(K)
    graph = [[] for _ in range(N+1)]  # 1-indexed
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        graph[a].append(b)
        graph[b].append(a)
    guards = []
    for _ in range(K):
        p_i, h_i = map(int, sys.stdin.readline().split())
        guards.append((h_i, p_i))

    max_stamina = [-1]*(N+1)  # 1-indexed
    import heapq
    heap = []
    for h_i, p_i in guards:
        heapq.heappush(heap, (-h_i, p_i))  # Use negative h_i for max-heap

    while heap:
        neg_stamina, u = heapq.heappop(heap)
        current_stamina = -neg_stamina
        if current_stamina <= max_stamina[u]:
            continue
        max_stamina[u] = current_stamina
        if current_stamina > 0:
            for v in graph[u]:
                if current_stamina-1 > max_stamina[v]:
                    heapq.heappush(heap, (-(current_stamina -1), v))

    guarded_vertices = [v for v in range(1,N+1) if max_stamina[v]>=0]
    G = len(guarded_vertices)
    print(G)
    print(' '.join(map(str, sorted(guarded_vertices))))

threading.Thread(target=main).start()