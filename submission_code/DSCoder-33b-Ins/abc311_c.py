# YOUR CODE HERE

import sys

def find_cycle(n, a):
    visited = [False] * (n + 1)
    cycle = []
    for i in range(1, n + 1):
        if not visited[i]:
            cycle.append(i)
            visited[i] = True
            j = a[i - 1]
            while j != i:
                cycle.append(j)
                visited[j] = True
                j = a[j - 1]
    return cycle

n = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))
cycle = find_cycle(n, a)
print(len(cycle))
print(' '.join(map(str, cycle)))