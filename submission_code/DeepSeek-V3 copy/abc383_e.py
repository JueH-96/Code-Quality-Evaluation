import sys
import heapq

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size + 1))
        self.size = [1] * (size + 1)
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        if self.size[xroot] < self.size[yroot]:
            xroot, yroot = yroot, xroot
        self.parent[yroot] = xroot
        self.size[xroot] += self.size[yroot]
        return True

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    K = int(data[idx+2])
    idx += 3
    edges = []
    for _ in range(M):
        u = int(data[idx])
        v = int(data[idx+1])
        w = int(data[idx+2])
        edges.append((w, u, v))
        idx += 3
    A = list(map(int, data[idx:idx+K]))
    idx += K
    B = list(map(int, data[idx:idx+K]))
    
    # Sort edges by weight
    edges.sort()
    
    # Initialize Union-Find
    uf = UnionFind(N)
    
    # To store the minimum f(A_i, B_i) for each pair
    # We need to find the minimum weight such that A_i and B_i are connected
    # So we can process edges in order and for each edge, union the two vertices
    # Then, for each pair (A_i, B_i), find the smallest weight where they are connected
    
    # To manage the pairs, we need to find for each A_i and B_i, the smallest weight where they are in the same set
    # We can use a list to store the current state of the pairs
    
    # Initialize the pairs
    pairs = list(zip(A, B))
    
    # To find the minimum weight for each pair, we can process the edges in order and for each edge, check if it connects any pair
    # We can use a priority queue to manage the pairs and their current minimum weight
    
    # Initialize a list to store the minimum weight for each pair
    min_weights = [float('inf')] * K
    
    # Iterate through the sorted edges
    for w, u, v in edges:
        uf.union(u, v)
        # Check all pairs to see if they are now connected
        for i in range(K):
            a, b = pairs[i]
            if uf.find(a) == uf.find(b):
                if min_weights[i] == float('inf'):
                    min_weights[i] = w
    
    # Now, we need to assign the B_i to A_i in such a way that the sum of min_weights is minimized
    # Since the min_weights are already the smallest possible for each pair, we can just sum them up
    # But we need to ensure that each B_i is assigned to exactly one A_i
    
    # Since the problem allows us to permute B, we can assign the smallest min_weight to the pair with the smallest A_i, etc.
    # But since the min_weights are already the smallest possible, we can just sum them up
    
    # So the answer is the sum of the min_weights
    print(sum(min_weights))

if __name__ == "__main__":
    main()