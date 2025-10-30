import sys
import heapq
from operator import itemgetter

def main():
    N, Q = map(int, sys.stdin.readline().split())
    query = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    query.sort(key=itemgetter(1))

    INF = 10**18
    edge = [INF]*(N+1)
    ans = 0

    for l, r, c in query:
        tmp = min(edge[l-1:r])
        if tmp < c:
            continue
        ans += c
        edge[l-1:r] = [c]*(r-l+1)

    if max(edge) == INF:
        print(-1)
    else:
        print(ans + sum(edge))

if __name__ == "__main__":
    main()