# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    NODES = N
    EDGES = [[] for _ in range(N)]
    for _ in range(M):
        U, V, W = sys.stdin.readline().split()
        U = int(U) - 1
        V = int(V) - 1
        W = int(W)
        EDGES[U].append((V, W))

    INF = 1 << 60
    NSTATE = 1 << N
    DP = [ [INF]*NSTATE for _ in range(N) ]

    for u in range(N):
        DP[u][1 << u] = 0

    for S in range(1, NSTATE):
        for u in range(N):
            if not (S & (1 << u)):
                continue
            curr_cost = DP[u][S]
            if curr_cost == INF:
                continue
            for v, w in EDGES[u]:
                S2 = S | (1 << v)
                if DP[v][S2] > curr_cost + w:
                    DP[v][S2] = curr_cost + w

    FULL = (1 << N) - 1
    ans = INF
    for u in range(N):
        if DP[u][FULL] < ans:
            ans = DP[u][FULL]
    if ans == INF:
        print("No")
    else:
        print(ans)