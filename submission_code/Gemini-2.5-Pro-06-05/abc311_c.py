import sys

# Although the solution is iterative, setting the recursion limit is a
# good practice for competitive programming problems with deep graph traversals,
# as a recursive solution would be a common alternative.
sys.setrecursionlimit(2 * 10**5 + 5)

def main():
    """
    Reads graph input, finds one cycle, and prints it to standard output.
    """
    # Read problem input from standard input
    try:
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle cases with empty or malformed input
        return

    # Convert the 1-based input to a 0-indexed adjacency array.
    # adj[i] stores the vertex that vertex i points to.
    adj = [x - 1 for x in a]

    # visited[i] becomes True once node i has been part of any traversal.
    # This prevents re-processing components of the graph.
    visited = [False] * n

    # Iterate through all vertices. If a vertex hasn't been visited,
    # it marks the start of a new component to explore.
    for i in range(n):
        if not visited[i]:
            # This component must contain a cycle, as every node has an out-degree of 1.
            
            # Trace the path starting from node i.
            path = []
            curr = i
            
            # Follow the path until we encounter a node that has already been visited.
            while not visited[curr]:
                visited[curr] = True
                path.append(curr)
                curr = adj[curr]
            
            # The first visited node we encounter (`curr`) must be on the current path.
            # Find where the cycle begins in the recorded path.
            start_index = path.index(curr)
            
            # The cycle is the suffix of the path starting from that index.
            cycle = path[start_index:]
            
            # Print the result as specified.
            print(len(cycle))
            # Convert vertices back to 1-based for the output.
            print(*(v + 1 for v in cycle))
            
            # We only need to find one cycle, so we are done.
            return

if __name__ == "__main__":
    main()