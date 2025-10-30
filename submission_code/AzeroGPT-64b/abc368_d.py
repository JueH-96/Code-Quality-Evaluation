from collections import defaultdict

def solve(tree, A, K):
    required = set(A)
    dp = [-1]*(len(tree)+1)
    distances = {}
    
    def find_depth(root, target, parent=None, depth=0):
        if root not in distances:
            distances[root] = {}
        if root == target:
            return depth
        for node in tree[root]:
            if node == parent: continue
            found = find_depth(node, target, root, depth+1)
            if found != -1:
                distances[root][node] = found - depth
                return found
        return -1
    
    def update_dp(root, required_nodes, parent=None):
        if required_nodes == 1:
            dp[root] = 1
            return dp[root]
        
        min_nodes = 1
        for child in tree[root]:
            if child == parent:
                continue
            
            if child in required_nodes:
                required_nodes.remove(child)
                distance = distances[root][child]
                nearest_required = min((d, node) for node, d in distances[child].items() if node in required_nodes)[1]
                required_nodes.add(nearest_required)

                update_dp(child, required_nodes, root)
                min_nodes += dp[child]
                if distance <= dp[child]: min_nodes -= 1

        dp[root] = min_nodes
        return min_nodes
    
    required_nodes = required.copy()
    required_nodes_df = []
    for required_node in required:
        for node in tree[required_node]:
            distances[required_node][node] = find_depth(node, required_node)
            required_nodes_df.append((required_node, node))
    required_nodes_df = list(set(required_nodes_df))
    
    for root, _ in required_nodes_df:
        update_dp(root, required_nodes.copy())
    
    return min(dp[required_node] for required_node in required)

n, K = map(int, input().split())
tree = defaultdict(list)
for _ in range(n-1):
    a, b = map(int, input().split())
    tree[a].append(b)
    tree[b].append(a)
required = list(map(int, input().split()))

print(solve(tree, required, K))