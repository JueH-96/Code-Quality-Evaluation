from collections import defaultdict, deque

def min_operations_to_delete_root(N, edges):
    # Create an adjacency list
    tree = defaultdict(list)
    for u, v in edges:
        tree[u].append(v)
        tree[v].append(u)

    # Initialize a queue with all leaf nodes except the root
    queue = deque()
    for node in range(2, N+1):
        if len(tree[node]) == 1:
            queue.append(node)

    # Count the number of operations
    operations = 0
    while queue:
        leaf = queue.popleft()
        operations += 1
        # Remove the leaf from the tree
        for neighbor in tree[leaf]:
            tree[neighbor].remove(leaf)
            # If the neighbor becomes a leaf, add it to the queue
            if len(tree[neighbor]) == 1 and neighbor != 1:
                queue.append(neighbor)
        # Clear the leaf's adjacency list
        tree[leaf] = []

    return operations

# Read input
N = int(input().strip())
edges = [tuple(map(int, input().strip().split())) for _ in range(N-1)]

# Solve the problem
print(min_operations_to_delete_root(N, edges))