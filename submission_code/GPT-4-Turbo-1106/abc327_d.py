def is_good_pair(N, M, A, B):
    # Create a dictionary to store the edges of the graph
    graph = {}
    for i in range(M):
        if A[i] not in graph:
            graph[A[i]] = set()
        if B[i] not in graph:
            graph[B[i]] = set()
        graph[A[i]].add(B[i])
        graph[B[i]].add(A[i])

    # Initialize the sequence X with -1 (unassigned)
    X = [-1] * (N + 1)

    # Helper function to perform DFS and assign values to X
    def dfs(node, value):
        # If the value is already assigned and it's different, return False
        if X[node] != -1 and X[node] != value:
            return False
        # If the value is already assigned and it's the same, return True
        if X[node] == value:
            return True
        # Assign the value to the node
        X[node] = value
        # Perform DFS on all connected nodes with the opposite value
        for neighbor in graph.get(node, []):
            if not dfs(neighbor, 1 - value):
                return False
        return True

    # Try to assign values to all nodes starting from each unvisited node
    for node in range(1, N + 1):
        if X[node] == -1:
            if not dfs(node, 0):
                return "No"
    return "Yes"

# Read input from stdin
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Check if the given pair of sequences is good and print the result
print(is_good_pair(N, M, A, B))