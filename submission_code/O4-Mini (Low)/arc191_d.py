import sys
import threading
from collections import deque

def main():
    import sys
    input = sys.stdin.readline
    N, M, S, T = map(int, input().split())
    # zero‐index
    S -= 1
    T -= 1
    
    G = [[] for _ in range(N)]
    direct_ST = False
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        G[u].append(v)
        G[v].append(u)
        if (u == S and v == T) or (u == T and v == S):
            direct_ST = True

    # If the graph is a tree, impossible to swap
    if M == N - 1:
        print(-1)
        return

    # BFS from S to get distance to T
    dist = [-1]*N
    dq = deque([S])
    dist[S] = 0
    while dq:
        u = dq.popleft()
        for w in G[u]:
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                dq.append(w)
    d = dist[T]
    if d < 0:
        # shouldn't happen in connected graph
        print(-1)
        return

    # If S and T are adjacent we need 3 moves, otherwise it's 2*d.
    if direct_ST:
        print(3)
    else:
        print(2 * d)

if __name__ == "__main__":
    main()