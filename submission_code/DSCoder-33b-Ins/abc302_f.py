# YOUR CODE HERE
from collections import defaultdict
from heapq import heappop, heappush

def solve():
    N, M = map(int, input().split())
    sets = defaultdict(list)
    for i in range(N):
        A = int(input())
        S = list(map(int, input().split()))
        for s in S:
            sets[s].append(i)
    if 1 not in sets or M not in sets:
        print(-1)
        return
    visited = [False]*N
    heap = [(0, i) for i in sets[1]]
    while heap:
        cost, i = heappop(heap)
        if visited[i]:
            continue
        visited[i] = True
        if i in sets[M]:
            print(cost)
            return
        for s in sets:
            if s in sets[i] and not visited[sets[s][0]]:
                heappush(heap, (cost+1, sets[s][0]))
    print(-1)

solve()