# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    from collections import deque

    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u-1].append(v-1)
        adj[v-1].append(u-1)
    W = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    # Step 1: Compute k_x for each node
    k = [0]*N
    sorted_neighbors = []
    for x in range(N):
        neighbors = sorted(adj[x], key=lambda y: W[y])
        sorted_neighbors.append(neighbors)
        total = 0
        cnt = 0
        for y in neighbors:
            if total + W[y] < W[x]:
                total += W[y]
                cnt +=1
            else:
                break
        k[x] = cnt

    # Step 2: Compute S(x) for each node
    # First, we need to have k_y for sorting
    # Since k is already computed, proceed
    S = [[] for _ in range(N)]
    for x in range(N):
        if k[x]==0:
            continue
        # Sort neighbors by k[y] descendingly
        neighbors = sorted(adj[x], key=lambda y: -k[y])
        S[x] = neighbors[:k[x]]

    # Step 3: BFS-like simulation
    counts = A[:]
    q = deque()
    for x in range(N):
        if counts[x] >0:
            q.append(x)
    res = 0
    while q:
        x = q.popleft()
        t = counts[x]
        res += t
        counts[x] =0
        for y in S[x]:
            if counts[y] ==0:
                q.append(y)
            counts[y] +=t
    print(res)

threading.Thread(target=main,).start()