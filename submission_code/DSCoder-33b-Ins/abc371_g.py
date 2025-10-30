# YOUR CODE HERE

import sys

def solve():
    n = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))

    p = [0] + p
    a = [0] + a

    visited = [False] * (n + 1)
    cycles = []

    for i in range(1, n + 1):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                cycle.append(a[j])
                visited[j] = True
                j = p[j]
            cycles.append(cycle)

    for cycle in cycles:
        cycle.sort()

    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        j = p[i]
        ans[j] = cycles[visited.index(True)][0]
        cycles[visited.index(True)].pop(0)

    print(*ans[1:])

solve()