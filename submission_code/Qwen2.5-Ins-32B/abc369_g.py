import sys
from collections import defaultdict

def dfs(node, parent, depth, tree, depths):
    depths.append(depth)
    for neighbor, weight in tree[node]:
        if neighbor != parent:
            dfs(neighbor, node, depth + weight, tree, depths)

def solve():
    N = int(input())
    tree = defaultdict(list)
    for _ in range(N-1):
        U, V, L = map(int, input().split())
        tree[U].append((V, L))
        tree[V].append((U, L))
    
    depths = []
    dfs(1, -1, 0, tree, depths)
    depths.sort()
    
    max_depth = max(depths)
    total_depth = sum(depths)
    num_vertices = len(depths)
    
    for K in range(1, N+1):
        if K == 1:
            print(total_depth * 2 - max_depth)
        elif K == 2:
            print(total_depth * 2)
        else:
            print(total_depth * 2 + (K-2) * max_depth)

solve()