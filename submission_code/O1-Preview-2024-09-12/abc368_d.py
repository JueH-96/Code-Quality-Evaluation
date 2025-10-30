# YOUR CODE HERE
import sys
import threading
sys.setrecursionlimit(1 << 25)

def main():
    N, K = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(N -1):
        A, B = map(int, sys.stdin.readline().split())
        adj[A].append(B)
        adj[B].append(A)
    V = list(map(int, sys.stdin.readline().split()))
    specified = [0] * (N +1)
    for v in V:
        specified[v] = 1
    to_include = [0] * (N + 1)
    def dfs(u, parent):
        need = specified[u]
        for v in adj[u]:
            if v != parent:
                need |= dfs(v, u)
        if need:
            to_include[u] = 1
        return need
    dfs(V[0], -1)
    result = sum(to_include)
    print(result)
threading.Thread(target=main).start()