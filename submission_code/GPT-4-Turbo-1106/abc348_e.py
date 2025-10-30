from collections import defaultdict

def dfs(tree, node, parent, depth, C, depth_sum, total_weight):
    depth_sum[node] = depth * C[node-1]
    total_weight[node] = C[node-1]
    
    for child in tree[node]:
        if child != parent:
            dfs(tree, child, node, depth + 1, C, depth_sum, total_weight)
            depth_sum[node] += depth_sum[child]
            total_weight[node] += total_weight[child]

def solve(N, edges, C):
    tree = defaultdict(list)
    for A, B in edges:
        tree[A].append(B)
        tree[B].append(A)
    
    depth_sum = [0] * (N + 1)
    total_weight = [0] * (N + 1)
    
    # Run DFS from node 1 to calculate depth_sum and total_weight for each node
    dfs(tree, 1, -1, 0, C, depth_sum, total_weight)
    
    # f(1) is already calculated as depth_sum[1]
    answer = depth_sum[1]
    
    # Now, use the values from the root to calculate f(v) for all other nodes
    def re_root(node, parent):
        nonlocal answer
        answer = min(answer, depth_sum[node])
        
        for child in tree[node]:
            if child != parent:
                # Before moving to the child, update the depth_sum values
                depth_sum[node] -= depth_sum[child]
                total_weight[node] -= total_weight[child]
                depth_sum[child] += depth_sum[node] + total_weight[node]
                total_weight[child] += total_weight[node]
                
                re_root(child, node)
                
                # After DFS, restore the original values
                total_weight[child] -= total_weight[node]
                depth_sum[child] -= depth_sum[node] + total_weight[node]
                total_weight[node] += total_weight[child]
                depth_sum[node] += depth_sum[child]
    
    re_root(1, -1)
    return answer

# Read input
N = int(input().strip())
edges = [tuple(map(int, input().strip().split())) for _ in range(N-1)]
C = list(map(int, input().strip().split()))

# Solve the problem
print(solve(N, edges, C))