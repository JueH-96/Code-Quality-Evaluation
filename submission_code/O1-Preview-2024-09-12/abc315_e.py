# YOUR CODE HERE
import sys
import threading

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]  # Using 1-based indexing
    for i in range(1, N+1):
        parts = sys.stdin.readline().split()
        if len(parts) == 0:
            parts = sys.stdin.readline().split()
        C_i = int(parts[0])
        P_i = list(map(int, parts[1:]))
        while len(P_i) < C_i:
            parts = sys.stdin.readline().split()
            P_i.extend(map(int, parts))
        adj[i] = P_i

    visited = [False] * (N+1)
    order = []
    def dfs(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                dfs(v)
        order.append(u)

    dfs(1)
    order = order[::-1]
    # Exclude book 1
    res = [str(book) for book in order if book != 1]
    print(' '.join(res))

threading.Thread(target=main).start()