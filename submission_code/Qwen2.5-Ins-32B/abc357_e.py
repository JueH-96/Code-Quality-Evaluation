import sys
from collections import defaultdict

def solve():
    N = int(input())
    a = list(map(int, input().split()))
    a = [x - 1 for x in a]

    visited = [False] * N
    reachable = [0] * N
    cycle_start = -1
    cycle_length = 0

    for i in range(N):
        if not visited[i]:
            path = []
            while not visited[i]:
                visited[i] = True
                path.append(i)
                i = a[i]

            if i in path:
                cycle_start = path.index(i)
                cycle_length = len(path) - cycle_start
                for j in range(cycle_start, len(path)):
                    reachable[path[j]] = cycle_length

    for i in range(N):
        if reachable[i] == 0:
            j = i
            while reachable[j] == 0:
                reachable[j] = -1
                j = a[j]
            cycle_length = reachable[j]
            while reachable[i] == -1:
                reachable[i] = cycle_length
                i = a[i]

    print(sum(reachable))

solve()