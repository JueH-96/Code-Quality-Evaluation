# YOUR CODE HERE
import sys
import collections

def main():
    import sys
    import collections

    input = sys.stdin.read().split()
    N = int(input[0])
    edges = []
    index = 1
    for i in range(N - 1):
        A = int(input[index])
        B = int(input[index + 1])
        edges.append((A, B))
        index += 2

    # Create adjacency list
    adj = collections.defaultdict(list)
    for A, B in edges:
        adj[A].append(B)
        adj[B].append(A)

    # Initialize leaves
    leaves = [node for node in adj if len(adj[node]) == 1]

    result = []

    # Perform N/2 operations
    for _ in range(N // 2):
        # Choose two leaves
        X = leaves.pop()
        Y = adj[X][0]
        adj[Y].remove(X)
        if len(adj[Y]) == 1:
            leaves.append(Y)
        
        # Find the farthest leaf from Y
        queue = collections.deque([(Y, 0)])
        visited = set([Y])
        farthest_node = Y
        farthest_distance = 0
        while queue:
            current, distance = queue.popleft()
            if distance > farthest_distance:
                farthest_distance = distance
                farthest_node = current
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        # Choose the farthest leaf from Y
        Z = farthest_node
        result.append((Z, X))
        
        # Remove Z and its neighbor
        W = adj[Z][0]
        adj[W].remove(Z)
        if len(adj[W]) == 1:
            leaves.append(W)
        del adj[Z]

    # Print the result
    for X, Y in result:
        print(X, Y)

if __name__ == "__main__":
    main()