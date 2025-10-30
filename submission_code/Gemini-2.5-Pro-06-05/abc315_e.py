import sys

def solve():
    """
    This function encapsulates the entire logic for solving the problem.
    It reads the input, builds a dependency graph, performs a topological sort
    on the subgraph required for book 1, and prints the result.
    """

    # For deep dependency chains, N can be up to 2*10^5.
    # We must increase Python's default recursion limit to avoid a RecursionError.
    sys.setrecursionlimit(2 * 10**5 + 5)

    try:
        # Read the total number of books from standard input.
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Handle cases with empty input, although not expected by problem constraints.
        return

    # We model the dependencies as a directed graph using an adjacency list.
    # adj[i] will store a list of prerequisites for book i.
    # Books are 1-indexed, so we create a list of size N+1.
    adj = [[] for _ in range(N + 1)]

    # Populate the adjacency list from the input.
    for i in range(1, N + 1):
        line = sys.stdin.readline().split()
        # The first element C_i is the count of prerequisites.
        # The actual prerequisites are the subsequent numbers on the line.
        # We handle the case where a book has no prerequisites (C_i = 0).
        if len(line) > 1:
            adj[i] = list(map(int, line[1:]))

    # 'visited' array tracks nodes that have been fully processed by the DFS.
    # A book is considered visited after all its prerequisites have been processed.
    visited = [False] * (N + 1)
    
    # 'path' will store the topological sort of the required books.
    # A post-order DFS traversal generates a valid reading order.
    path = []

    def dfs(u):
        """
        Performs a recursive Depth-First Search (DFS) from node u.
        This function implements a post-order traversal, which is a standard
        way to compute a topological sort.

        Args:
            u (int): The current book (node) to visit.
        """
        visited[u] = True
        # Recursively visit all unvisited prerequisites of the current book.
        for v in adj[u]:
            if not visited[v]:
                dfs(v)
        # After all prerequisites are handled, add the current book to the path.
        path.append(u)

    # To read book 1, we must read all its transitive prerequisites.
    # A DFS starting from book 1 will naturally traverse this required set of books.
    dfs(1)

    # The 'path' list now contains all required books, with book 1 at the end.
    # The order of elements in 'path' is a valid reading sequence.
    # We need to print the prerequisites, which are all elements except the last one.
    result_to_print = path[:-1]

    # Print the book numbers in the calculated reading order, separated by spaces.
    print(*result_to_print)

# Run the main solver function.
solve()