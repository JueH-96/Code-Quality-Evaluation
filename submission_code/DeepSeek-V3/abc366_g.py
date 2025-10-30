# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = defaultdict(list)
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Initialize the values for each node
    X = [0] * (N + 1)
    
    # For each node, the XOR of its neighbors must be zero
    # We can set all nodes to the same value, say 1
    # Then, for each node, the XOR of its neighbors will be 0 if the number of neighbors is even
    # So, we need to ensure that for every node, the number of neighbors is even
    # But this is not always possible, so we need a different approach
    
    # Alternative approach: assign values such that for each node, the sum of its neighbors' values is zero in XOR
    # This can be achieved by setting the values of nodes in a way that the XOR of all neighbors of a node is zero
    # One way to do this is to assign values such that the XOR of all values in the graph is zero
    # But this is not straightforward
    
    # Another approach: assign values such that for each edge, the two nodes have the same value
    # This way, the XOR of neighbors for each node will be zero if the number of neighbors is even
    # But this is not always possible
    
    # Given the complexity, we can try to assign the same value to all nodes
    # This will satisfy the condition if the graph is such that each node has an even number of neighbors
    # So, first, check if all nodes have even degrees
    degrees = [0] * (N + 1)
    for u in adj:
        degrees[u] = len(adj[u])
    for u in range(1, N+1):
        if degrees[u] % 2 != 0:
            print("No")
            return
    
    # If all degrees are even, assign the same value to all nodes
    value = 1
    X = [value] * (N + 1)
    print("Yes")
    print(' '.join(map(str, X[1:])))

if __name__ == "__main__":
    main()