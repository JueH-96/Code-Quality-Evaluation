import sys

# Set higher recursion depth limit
# To handle potentially deep dependency chains up to N = 2*10^5.
# The maximum depth of the recursion could potentially be N in worst cases like a long chain.
try:
    # Set recursion depth to a value slightly larger than the maximum possible N
    sys.setrecursionlimit(2 * 10**5 + 10) 
except Exception: 
    # If setting recursion limit fails (e.g., due to OS restrictions or other issues),
    # proceed with the default limit. This might lead to RecursionError for deep inputs.
    pass 

def solve():
    # Read the number of books
    N = int(sys.stdin.readline())

    # Adjacency list for the reversed dependency graph (G^R)
    # `adj[i]` will store the list of indices of books required *before* reading book i+1.
    # We use 0-based indexing internally for nodes (0 to N-1). Book k corresponds to index k-1.
    adj = [[] for _ in range(N)] 

    # Read input for each book and build the reversed graph G^R
    for i in range(N):
        # Read the line containing C_i and the prerequisites P_{i,j}
        line = list(map(int, sys.stdin.readline().split()))
        
        # The count of prerequisites C_i is line[0]. The prerequisites P_{i,j} start from line[1].
        # Check if there are any prerequisites (C_i > 0)
        if len(line) > 1: 
            prerequisites = line[1:]
            for p_book_num in prerequisites:
                # Book i+1 (represented by node i) requires book p_book_num (represented by node p_book_num-1).
                # In the reversed graph G^R, this corresponds to a directed edge from node i to node p_book_num-1.
                # This edge means "to process node i, we must first process node p_book_num-1".
                adj[i].append(p_book_num - 1) # Store the 0-based index of the prerequisite book

    # `visited[k]` will be True if book k+1 (node k) has been visited during the DFS traversal.
    visited = [False] * N
    
    # `read_order_indices` will store the 0-based indices of the required books.
    # The books are added in post-order traversal, which corresponds to a valid reading sequence.
    read_order_indices = []

    # Recursive Depth First Search function
    def dfs(u_idx):
        """ Performs DFS on the reversed graph G^R starting from node u_idx.
            Identifies all required books for the initial target book (book 1 / node 0)
            and populates read_order_indices in a valid topological sort order (post-order). """
        
        # Mark the current node as visited
        visited[u_idx] = True
        
        # Explore all neighbors (prerequisites) in the reversed graph
        for v_idx in adj[u_idx]:
            # If a prerequisite node hasn't been visited yet, recursively call DFS on it.
            if not visited[v_idx]:
                dfs(v_idx)
        
        # After all prerequisites (descendants in the DFS tree rooted at u_idx within G^R)
        # have been visited and processed (their recursive calls returned), 
        # append the current node index u_idx to the `read_order_indices` list.
        # This post-order addition ensures that a book is added only after all its dependencies
        # within the required set R have been added.
        read_order_indices.append(u_idx)

    # The problem asks for the minimum set of books required to read book 1.
    # Book 1 corresponds to index 0 in our 0-based indexing.
    # We start DFS from node 0. This traversal will visit exactly the set R of required books:
    # book 1 itself and all its direct and indirect prerequisites.
    # The `visited` array tracks nodes in R. The `read_order_indices` list stores them in a valid reading order.
    
    # Initiate the DFS traversal starting from node 0 (representing book 1).
    # The check `if not visited[0]` is standard for DFS on potentially disconnected graphs,
    # ensuring each component is visited once. Here, we are only interested in the component
    # related to book 1, so starting from node 0 is sufficient. The check is harmless.
    if not visited[0]: 
       dfs(0) 

    # After the DFS completes, `read_order_indices` contains the 0-based indices of all books in R.
    # The order is a topological sort, meaning it's a valid sequence to read the books.
    # Because of the post-order traversal property, the initial node (book 1 / index 0) 
    # will be the last element added to the list.
    
    # The problem asks to print the required books *excluding* book 1.
    # We can achieve this by taking all elements of the list except the last one.
    result_indices = read_order_indices[:-1] # Slicing creates a new list without the last element.
    
    # Convert the 0-based indices back to 1-based book numbers, which is the required output format.
    result_book_nums = [idx + 1 for idx in result_indices]
    
    # Print the resulting list of book numbers, separated by spaces.
    # The problem statement guarantees C1 >= 1, meaning book 1 has at least one prerequisite.
    # This ensures that the set R contains more than just book 1, so `result_book_nums` will not be empty.
    # The `*` operator unpacks the list elements as arguments to print, which handles space separation.
    print(*(result_book_nums))

# Execute the main solver function when the script is run.
solve()