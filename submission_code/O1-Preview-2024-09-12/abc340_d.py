# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import heapq
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = []
    B = []
    X = []
    for _ in range(N-1):
        a,b,x = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        X.append(x-1)  # zero-based indexing

    # Build the graph
    from collections import defaultdict

    adj = defaultdict(list)
    for i in range(N-1):
        adj[i].append((i+1, A[i]))
        adj[i].append((X[i], B[i]))

    dist = [float('inf')] * N
    dist[0] = 0
    hq = []
    heapq.heappush(hq, (0, 0))  # (cost, node)

    while hq:
        cost, u = heapq.heappop(hq)
        if dist[u] < cost:
            continue
        for v, w in adj[u]:
            if cost + w < dist[v]:
                dist[v] = cost + w
                heapq.heappush(hq, (dist[v], v))

    print(dist[N-1])

if __name__ == "__main__":
    threading.Thread(target=main).start()