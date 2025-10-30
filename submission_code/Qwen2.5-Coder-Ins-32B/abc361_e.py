# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N-1):
        A = int(data[index])
        B = int(data[index+1])
        C = int(data[index+2])
        edges.append((C, A, B))
        index += 3
    
    # Kruskal's algorithm to find the Minimum Spanning Tree (MST)
    parent = list(range(N + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootY] = rootX
            return True
        return False
    
    # Sort edges by weight
    edges.sort()
    
    mst_weight = 0
    for weight, A, B in edges:
        if union(A, B):
            mst_weight += weight
    
    # The minimum travel distance is twice the weight of the MST
    print(2 * mst_weight)

if __name__ == "__main__":
    main()