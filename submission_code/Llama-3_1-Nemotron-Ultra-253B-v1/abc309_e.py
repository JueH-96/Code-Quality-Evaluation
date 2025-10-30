import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    p = list(map(int, sys.stdin.readline().split()))
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        parent = p[i-2]
        children[parent].append(i)
    depth = [0] * (N + 1)
    for i in range(2, N+1):
        depth[i] = depth[p[i-2]] + 1
    max_own = [-float('inf')] * (N + 1)
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        val = depth[x] + y
        if val > max_own[x]:
            max_own[x] = val
    count = 0
    queue = deque()
    root_max = max_own[1]
    if root_max >= depth[1]:
        count += 1
    queue.append((1, root_max))
    while queue:
        u, current_max = queue.popleft()
        for v in children[u]:
            new_max = max(current_max, max_own[v])
            if new_max >= depth[v]:
                count += 1
            queue.append((v, new_max))
    print(count)

if __name__ == '__main__':
    main()