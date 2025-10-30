import collections
import sys

def solve():
    N = int(sys.stdin.readline())

    # orig_graph_adj[i] stores the list of books that book 'i' requires to be read before it.
    # So, there's a directed edge from each book in orig_graph_adj[i] to 'i'.
    orig_graph_adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        line = list(map(int, sys.stdin.readline().split()))
        # C_i is line[0], prerequisites are line[1:]
        orig_graph_adj[i].extend(line[1:])

    # Step 1: Identify all books that must be read to eventually read book 1.
    # This is done by performing a BFS starting from book 1 and traversing
    # to its direct prerequisites, then their prerequisites, and so on.
    # This effectively finds all nodes reachable from book 1 by following reverse edges
    # (where a reverse edge means going from a book to its prerequisite).
    
    required_books_set = set()
    q = collections.deque()

    # Start with book 1, as it's the target book
    q.append(1)
    required_books_set.add(1)

    while q:
        current_book = q.popleft()
        # For each prerequisite of the current_book (i.e., books that current_book needs)
        for prereq_book in orig_graph_adj[current_book]:
            if prereq_book not in required_books_set:
                required_books_set.add(prereq_book)
                q.append(prereq_book)

    # The set 'required_books_set' now contains book 1 and all its necessary prerequisites.
    # We need to output the books *excluding* book 1.
    books_to_sort = [book for book in required_books_set if book != 1]

    # If no books are required other than book 1 (e.g., C_1 = 0, but problem states C_1 >= 1)
    if not books_to_sort:
        sys.stdout.write("
") # Print empty line if no books
        return

    # Step 2: Perform a topological sort on the 'books_to_sort' subset.
    # This ensures that if book A is a prerequisite for book B, A appears before B in the output.
    
    # subgraph_in_degree[book] will store the count of prerequisites (within books_to_sort) for 'book'.
    subgraph_in_degree = {book: 0 for book in books_to_sort}
    # subgraph_adj[book] will store books that 'book' is a prerequisite for (within books_to_sort).
    subgraph_adj = {book: [] for book in books_to_sort}

    # Iterate through all books that are in our target output set (i.e., books_to_sort).
    for book_u in books_to_sort:
        # orig_graph_adj[book_u] lists all prerequisites for book_u.
        # These are books `prereq_v` such that `prereq_v -> book_u` is an edge.
        for prereq_v in orig_graph_adj[book_u]:
            # If this prerequisite 'prereq_v' is also in our 'books_to_sort' set,
            # then it's a dependency relevant to the subgraph we're sorting.
            if prereq_v in books_to_sort:
                # Add a directed edge from prereq_v to book_u in our subgraph representation
                subgraph_adj[prereq_v].append(book_u)
                # Increment the in-degree for book_u, as it has an incoming edge from prereq_v
                subgraph_in_degree[book_u] += 1
            # Note: If prereq_v is book 1, or any other book not in `books_to_sort`,
            # it means it's not part of the set of books to be outputted.
            # We don't count it as an in-degree for the purpose of sorting `books_to_sort`.

    # Initialize a queue for Kahn's algorithm (topological sort)
    q = collections.deque()
    for book in books_to_sort:
        if subgraph_in_degree[book] == 0:
            q.append(book)

    result_order = []
    while q:
        u = q.popleft()
        result_order.append(u)

        # For each book 'v' that 'u' is a prerequisite for (i.e., u -> v exists in the subgraph)
        for v in subgraph_adj[u]:
            subgraph_in_degree[v] -= 1 # Decrement in-degree of 'v'
            if subgraph_in_degree[v] == 0:
                q.append(v) # If 'v' now has no remaining prerequisites, add it to the queue
    
    # Print the resulting topological order, space-separated
    sys.stdout.write(" ".join(map(str, result_order)) + "
")

solve()