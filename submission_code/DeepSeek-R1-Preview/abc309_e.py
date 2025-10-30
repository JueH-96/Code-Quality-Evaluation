import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    n, m = map(int, sys.stdin.readline().split())
    p = list(map(int, sys.stdin.readline().split()))
    
    # Build parent array
    parent = [0] * (n + 1)
    parent[1] = 0  # Root has no parent
    for i in range(2, n + 1):
        parent[i] = p[i - 2]
    
    # Compute depth for each node
    depth = [0] * (n + 1)
    for i in range(2, n + 1):
        depth[i] = depth[parent[i]] + 1
    
    # Read M insurances and compute max_cover
    max_cover = [-float('inf')] * (n + 1)
    for _ in range(m):
        x, y = map(int, sys.stdin.readline().split())
        current_val = depth[x] + y
        if current_val > max_cover[x]:
            max_cover[x] = current_val
    
    # Group nodes by depth
    depth_groups = defaultdict(list)
    for u in range(1, n + 1):
        depth_groups[depth[u]].append(u)
    
    # Process nodes in order of increasing depth
    depths = sorted(depth_groups.keys())
    max_current = [-float('inf')] * (n + 1)
    count = 0
    
    for d in depths:
        for u in depth_groups[d]:
            if u == 1:
                mc = max_cover[u]
            else:
                mc = max(max_cover[u], max_current[parent[u]])
            max_current[u] = mc
            if mc >= depth[u]:
                count += 1
                
    print(count)

if __name__ == "__main__":
    main()