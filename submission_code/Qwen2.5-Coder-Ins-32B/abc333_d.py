# YOUR CODE HERE
import sys
from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    edges = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(N-1)]
    
    # Create adjacency list
    adj = [[] for _ in range(N+1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Initialize queue with all leaves
    queue = deque([i for i in range(1, N+1) if len(adj[i]) == 1])
    operations = 0
    
    while queue:
        size = len(queue)
        for _ in range(size):
            v = queue.popleft()
            for neighbor in adj[v]:
                adj[neighbor].remove(v)
                if len(adj[neighbor]) == 1:
                    queue.append(neighbor)
        if queue:
            operations += 1
    
    print(operations)

if __name__ == "__main__":
    main()