import sys
import sys
import sys
from collections import deque
def solve():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    idx = 1
    prereq = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        C_i = int(N_and_rest[idx])
        idx +=1
        prereq[i] = list(map(int, N_and_rest[idx:idx+C_i]))
        idx += C_i
    # Find S via DFS
    visited = [False] * (N+1)
    stack = [1]
    visited[1] = True
    while stack:
        u = stack.pop()
        for p in prereq[u]:
            if not visited[p]:
                visited[p] = True
                stack.append(p)
    S = set()
    for i in range(1, N+1):
        if visited[i] and i !=1:
            S.add(i)
    # Build in_degree and adj_s
    in_degree = {u:0 for u in S}
    adj_s = {u:[] for u in S}
    for u in S:
        for p in prereq[u]:
            if p in S:
                in_degree[u] +=1
                adj_s[p].append(u)
    # Kahn's algorithm
    queue = deque()
    for u in S:
        if in_degree[u] ==0:
            queue.append(u)
    result = []
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in adj_s[u]:
            in_degree[v] -=1
            if in_degree[v]==0:
                queue.append(v)
    print(' '.join(map(str, result)))