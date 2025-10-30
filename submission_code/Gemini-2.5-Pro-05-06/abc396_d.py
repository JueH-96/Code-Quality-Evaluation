import sys

class Solver:
    def __init__(self, N, M, edges):
        self.N = N
        self.adj = [[] for _ in range(N)]
        for u, v, w in edges:
            # Adjust to 0-indexed vertices for array indexing and bitmasks
            u_idx, v_idx = u - 1, v - 1
            self.adj[u_idx].append((v_idx, w))
            self.adj[v_idx].append((u_idx, w)) # Graph is undirected

        # Initialize min_total_xor. Edge labels w_i are < 2^60.
        # The XOR sum of such labels will also be < 2^60.
        # (1 << 60) serves as a value larger than any possible valid XOR sum.
        self.min_total_xor = (1 << 60) 
        
        # Store the 0-indexed target node (vertex N)
        self.target_node_idx = N - 1

    def dfs(self, u_idx, current_xor, visited_mask):
        # Base case: If the current vertex is the target vertex N (N-1 in 0-indexing)
        if u_idx == self.target_node_idx:
            # A path from start to target has been found.
            # Update the minimum XOR sum if the current path's sum is smaller.
            if current_xor < self.min_total_xor:
                self.min_total_xor = current_xor
            return # End exploration for this path

        # Recursive step: Explore neighbors of u_idx
        for v_idx, weight in self.adj[u_idx]:
            # Check if vertex v_idx has already been visited in the current path.
            # (visited_mask & (1 << v_idx)) is non-zero if v_idx's bit is set in the mask.
            if not (visited_mask & (1 << v_idx)):
                # If v_idx is not visited, recursively call DFS for v_idx.
                # Update current_xor by XORing with the edge weight.
                # Update visited_mask by setting the bit for v_idx.
                self.dfs(v_idx, current_xor ^ weight, visited_mask | (1 << v_idx))

    def solve(self):
        # Start DFS from vertex 1 (0-indexed as 0).
        # Initial XOR sum is 0 (no edges traversed yet).
        # Initial visited_mask has only the bit for vertex 0 set: (1 << 0).
        self.dfs(0, 0, (1 << 0))
        
        # After DFS completes, self.min_total_xor will hold the minimum XOR sum.
        # The problem guarantees a connected graph and N >= 2, so a path exists.
        return self.min_total_xor

def main():
    # Read N (number of vertices) and M (number of edges)
    N, M = map(int, sys.stdin.readline().split())
    
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((u, v, w))

    # Create solver instance and run the algorithm
    solver = Solver(N, M, edges)
    result = solver.solve()
    
    # Print the result
    print(result)

if __name__ == '__main__':
    main()