import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    operations = []
    for _ in range(Q):
        L = int(data[idx])
        R = int(data[idx+1])
        C = int(data[idx+2])
        operations.append((L, R, C))
        idx += 3
    
    # Initialize the parent array for Union-Find
    parent = [i for i in range(N + Q + 1)]
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        parent[v_root] = u_root
        return True
    
    # To manage the intervals, we can use a list to represent the connected components
    # We need to find the minimal cost to connect all components
    # We can treat the operations as edges between the new vertex and the range [L, R]
    # We need to find the minimal cost to connect all vertices
    
    # We can use a priority queue to select the edges with the smallest cost first
    # We need to manage the intervals efficiently
    
    # Create a list of all possible edges
    edges = []
    for i in range(Q):
        L, R, C = operations[i]
        edges.append((C, N + i + 1, L))
        edges.append((C, N + i + 1, R))
    
    # Sort edges by cost
    edges.sort()
    
    # Initialize the MST cost
    mst_cost = 0
    # Initialize the number of connected components
    num_components = N + Q
    # Iterate through the edges and add them to the MST if they connect different components
    for cost, u, v in edges:
        if union(u, v):
            mst_cost += cost
            num_components -= 1
            if num_components == 1:
                break
    
    # Check if the graph is connected
    if num_components != 1:
        print(-1)
    else:
        print(mst_cost)

if __name__ == "__main__":
    main()