import sys
from collections import defaultdict
from heapq import heappush, heappop

def main():
    N, Q = map(int, sys.stdin.readline().split())
    graph = defaultdict(list)
    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        if query[0] == 1:
            u, v = query[1], query[2]
            heappush(graph[u], -v)
            heappush(graph[v], -u)
        else:
            v, k = query[1], query[2]
            if len(graph[v]) < k:
                print(-1)
            else:
                print(-heappop([heappop(graph[v]) for _ in range(k)]))

if __name__ == "__main__":
    main()