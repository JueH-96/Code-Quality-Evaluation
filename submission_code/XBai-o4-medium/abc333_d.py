import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    children = [[] for _ in range(N+1)]
    visited = [False] * (N + 1)
    q = deque()
    q.append(1)
    visited[1] = True
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                children[u].append(v)
                q.append(v)
    
    steps = [0] * (N + 1)
    stack = [(1, False)]
    
    while stack:
        u, flag = stack.pop()
        if not flag:
            stack.append((u, True))
            for child in reversed(children[u]):
                stack.append((child, False))
        else:
            if u == 1:
                sum_children = 0
                max_child = 0
                for child in children[u]:
                    sum_children += steps[child]
                    if steps[child] > max_child:
                        max_child = steps[child]
                steps[u] = sum_children - max_child + 1
            else:
                sum_children = 0
                for child in children[u]:
                    sum_children += steps[child]
                steps[u] = sum_children + 1
    
    print(steps[1])

if __name__ == '__main__':
    main()