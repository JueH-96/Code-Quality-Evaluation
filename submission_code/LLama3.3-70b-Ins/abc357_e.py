from sys import stdin, stdout

def count_reachable_pairs(n, edges):
    """
    Count the number of pairs of vertices (u, v) such that vertex v is reachable from vertex u.

    Args:
    n (int): The number of vertices in the graph.
    edges (list): A list of edges where edges[i] is the vertex that vertex i points to.

    Returns:
    int: The number of pairs of vertices (u, v) such that vertex v is reachable from vertex u.
    """
    # Create a list to store the number of vertices reachable from each vertex
    reachable_counts = [0] * n

    # Iterate over each vertex
    for i in range(n):
        # Initialize a set to keep track of visited vertices
        visited = set()
        # Initialize a stack with the current vertex
        stack = [i]

        # Perform a depth-first search
        while stack:
            vertex = stack.pop()
            # If the vertex has not been visited before
            if vertex not in visited:
                # Mark the vertex as visited
                visited.add(vertex)
                # Increment the count of reachable vertices for the current vertex
                reachable_counts[i] += 1
                # Add the next vertex to the stack
                stack.append(edges[vertex] - 1)

    # Calculate the total number of pairs of vertices (u, v) such that vertex v is reachable from vertex u
    total_pairs = sum(reachable_counts)

    return total_pairs

def main():
    # Read the number of vertices
    n = int(stdin.readline().strip())
    # Read the edges
    edges = list(map(int, stdin.readline().strip().split()))

    # Count the number of pairs of vertices (u, v) such that vertex v is reachable from vertex u
    total_pairs = count_reachable_pairs(n, edges)

    # Print the result
    stdout.write(str(total_pairs) + "
")

if __name__ == "__main__":
    main()