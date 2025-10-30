import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v, l = map(int, sys.stdin.readline().split())
        edges[u].append((v, l))
        edges[v].append((u, l))
    
    # Compute path sums using BFS
    from collections import deque
    path_sum = [0] * (N + 1)
    visited = [False] * (N + 1)
    q = deque()
    q.append(1)
    visited[1] = True
    while q:
        u = q.popleft()
        for v, l in edges[u]:
            if not visited[v]:
                visited[v] = True
                path_sum[v] = path_sum[u] + l
                q.append(v)
    
    # Collect all path sums except 1 (since 1's path sum is 0)
    sums = []
    for u in range(2, N + 1):
        sums.append(path_sum[u])
    
    # Sort in descending order
    sums.sort(reverse=True)
    
    # For each K from 1 to N-1, compute the sum of the first K sums
    res = []
    current_sum = 0
    for k in range(1, N):
        current_sum += sums[k - 1]
        res.append(str(current_sum * 2))
    
    print('
'.join(res))

if __name__ == '__main__':
    main()