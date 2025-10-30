def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x == root_y:
        return False
    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    else:
        parent[root_y] = root_x
        if rank[root_x] == rank[root_y]:
            rank[root_x] += 1
    return True

def main():
    N, Q = map(int, input().split())
    
    operations = []
    for i in range(1, Q+1):
        L, R, C = map(int, input().split())
        operations.append((C, L, R, N+i))
    
    # Sort operations by cost
    operations.sort()
    
    parent = list(range(N+Q+1))
    rank = [0] * (N+Q+1)
    mst_cost = 0
    mst_edges = 0
    
    for C, L, R, u in operations:
        u_comp = find(parent, u)
        
        # Keep track of components we've already connected to
        connected_components = set()
        
        for j in range(L, R+1):
            j_comp = find(parent, j)
            
            # Check if the component of j is not already connected to u
            if j_comp != u_comp and j_comp not in connected_components:
                connected_components.add(j_comp)
                union(parent, rank, u, j)
                mst_cost += C
                mst_edges += 1
        
        if mst_edges == N+Q-1:
            break
    
    # Check if the graph is connected
    if mst_edges == N+Q-1:
        return mst_cost
    else:
        return -1

if __name__ == "__main__":
    result = main()
    print(result)