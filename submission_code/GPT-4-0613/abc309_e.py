import sys
from operator import itemgetter
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    P = [0] + P
    G = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        G[P[i-1]].append(i)
    I = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]
    I.sort(key=itemgetter(1), reverse=True)
    I = deque(I)
    Q = deque()
    D = [0]*(N+1)
    covered = [False]*(N+1)
    ans = 0
    while I:
        x, y = I.popleft()
        if D[x] or covered[x]:
            continue
        Q.append((x, y))
        while Q:
            v, d = Q.popleft()
            if D[v] or covered[v]:
                continue
            D[v] = d
            covered[v] = True
            ans += 1
            if d > 1:
                for u in G[v]:
                    Q.append((u, d-1))
    print(ans)

if __name__ == "__main__":
    main()