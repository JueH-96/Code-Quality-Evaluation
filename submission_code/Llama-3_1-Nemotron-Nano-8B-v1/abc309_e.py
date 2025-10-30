import sys
from sys import stdin

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, stdin.readline().split())
    parent = [0] * (N + 1)  # 1-based
    if N >= 2:
        p = list(map(int, stdin.readline().split()))
        for i in range(2, N+1):
            parent[i] = p[i-2]
    else:
        # N=1, but according to constraints N >=2
        pass
    
    # Compute depth for each node
    depth = [0] * (N + 1)
    for i in range(2, N+1):
        depth[i] = depth[parent[i]] + 1
    
    max_y = [-float('inf')] * (N + 1)
    for _ in range(M):
        x, y = map(int, stdin.readline().split())
        if y > max_y[x]:
            max_y[x] = y
    
    # Compute max_cover
    max_cover = [0] * (N + 1)
    max_cover[1] = depth[1] + max_y[1]
    for i in range(2, N+1):
        candidates = [max_cover[parent[i]]]
        if max_y[i] != -float('inf'):
            candidates.append(depth[i] + max_y[i])
        max_cover[i] = max(candidates)
    
    # Count the number of nodes where depth[i] <= max_cover[i]
    count = 0
    for i in range(1, N+1):
        if depth[i] <= max_cover[i]:
            count += 1
    print(count)

if __name__ == '__main__':
    main()