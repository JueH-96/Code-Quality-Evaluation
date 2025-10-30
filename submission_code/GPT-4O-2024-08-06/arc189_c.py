# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1]) - 1  # Convert to 0-indexed
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:2*N+2]))
    P = list(map(lambda x: int(x) - 1, data[2*N+2:3*N+2]))  # Convert to 0-indexed
    Q = list(map(lambda x: int(x) - 1, data[3*N+2:4*N+2]))  # Convert to 0-indexed
    
    # Union-Find (Disjoint Set) implementation
    parent = list(range(N))
    rank = [0] * N
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1
    
    # Create the union-find structure for both P and Q
    for i in range(N):
        union(i, P[i])
        union(i, Q[i])
    
    # Check if all components can be resolved to X
    component_to_boxes = {}
    for i in range(N):
        root = find(i)
        if root not in component_to_boxes:
            component_to_boxes[root] = []
        component_to_boxes[root].append(i)
    
    # Check each component
    operations = 0
    for component in component_to_boxes.values():
        if X in component:
            # We can resolve this component to X
            # Count the number of operations needed
            # We need to move all balls to X, so we need |component| - 1 operations
            operations += len(component) - 1
        else:
            # If X is not in the component, it's impossible
            print(-1)
            return
    
    print(operations)