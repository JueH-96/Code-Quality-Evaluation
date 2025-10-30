# YOUR CODE HERE

import sys

def solve():
    N = int(input())
    a = list(map(int, input().split()))
    a = [0] + a
    visited = [False] * (N + 1)
    cycles = []
    for i in range(1, N + 1):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                cycle.append(j)
                visited[j] = True
                j = a[j]
            cycles.append(cycle)
    ans = 0
    for cycle in cycles:
        ans += len(cycle) ** 2
    print(ans)

solve()