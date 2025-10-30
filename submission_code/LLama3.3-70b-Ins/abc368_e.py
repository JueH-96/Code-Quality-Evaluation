import sys
from collections import defaultdict, deque

def solve():
    N, M, X1 = map(int, sys.stdin.readline().split())
    A = []
    B = []
    S = []
    T = []
    for _ in range(M):
        a, b, s, t = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        S.append(s)
        T.append(t)

    graph = defaultdict(list)
    for i in range(M):
        for j in range(M):
            if B[i] == A[j] and T[i] <= S[j]:
                graph[i].append(j)

    X = [0] * M
    X[0] = X1
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if X[v] < X[u] + T[u] - S[v]:
                X[v] = X[u] + T[u] - S[v]
                queue.append(v)

    print(*X[1:])

if __name__ == "__main__":
    solve()