def get_parent(x):
    return x // 2

def get_children(x, n):
    children = []
    if 2*x <= n:
        children.append(2*x)
    if 2*x + 1 <= n:
        children.append(2*x + 1)
    return children

def get_distance_count(n, x, k):
    if k == 0:
        return 1 if x <= n else 0
        
    if k == 1:
        count = 0
        # Check parent
        if x > 1:
            count += 1
        # Check children
        if 2*x <= n:
            count += 1
        if 2*x + 1 <= n:
            count += 1
        return count
        
    # For k > 1
    # First get nodes at distance k-1
    prev_nodes = set()
    curr_nodes = {x}
    for _ in range(k):
        next_nodes = set()
        for node in curr_nodes:
            # Add parent
            if node > 1:
                parent = get_parent(node)
                if parent not in prev_nodes and parent not in curr_nodes:
                    next_nodes.add(parent)
            # Add children
            for child in get_children(node, n):
                if child not in prev_nodes and child not in curr_nodes:
                    next_nodes.add(child)
        prev_nodes = curr_nodes
        curr_nodes = next_nodes
        
    return len(curr_nodes)

T = int(input())
for _ in range(T):
    N, X, K = map(int, input().split())
    print(get_distance_count(N, X, K))