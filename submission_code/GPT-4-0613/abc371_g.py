import sys
from heapq import heapify, heappop, heappush

def main():
    n = int(sys.stdin.readline())
    p = list(map(int, sys.stdin.readline().split()))
    a = list(map(int, sys.stdin.readline().split()))
    p = [i-1 for i in p]
    a = [(a[i], i) for i in range(n)]
    a.sort()
    ans = [0]*n
    used = [False]*n
    g = [[] for _ in range(n)]
    for i in range(n):
        g[p[i]].append(i)
    for _, i in a:
        heap = [i]
        heapify(heap)
        while heap:
            v = heappop(heap)
            if used[v]:
                continue
            used[v] = True
            ans[v] = a[i][0]
            for to in g[v]:
                if not used[to]:
                    heappush(heap, to)
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()