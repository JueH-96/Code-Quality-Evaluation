import sys
from collections import defaultdict, deque

# Read input
N = int(sys.stdin.readline())

# Store prerequisites: prerequisites[i] is a list of books that must be read before book i
prerequisites = [[] for _ in range(N + 1)]
for i in range(1, N + 1):
    line = list(map(int, sys.stdin.readline().split()))
    C_i = line[0]
    prereqs_i = line[1:]
    prerequisites[i] = prereqs_i

# Step 1 & 2: Build graph G' (edges i -> p if p is a prerequisite for i)
# Find set R (required books excluding 1) using iterative DFS on G'
# The graph G' is implicitly represented by the prerequisites list. An edge i -> p exists if p is in prerequisites[i].

required_books_set = set()
stack_g_prime = [1] # Start DFS from book 1 in G'
visited_g_prime_set = set()

while stack_g_prime:
    book_id = stack_g_prime.pop() # Use pop for stack behavior (LIFO)

    if book_id in visited_g_prime_set:
        continue
    visited_g_prime_set.add(book_id)

    if book_id != 1:
        required_books_set.add(book_id)

    # Neighbors of book_id in G' are its prerequisites
    # The order of adding to stack affects DFS traversal order but not the final set of visited nodes.
    # No need to sort prerequisites[book_id] here.
    for prereq in prerequisites[book_id]:
         # Check if prereq is already visited before adding to stack is an optimization
         # Although the main check when popping handles cycles, pre-checking avoids stack growth.
         # Given it's a DAG (implied by "possible to read all books"), pre-check is not strictly necessary for correctness but can improve performance slightly.
         if prereq not in visited_g_prime_set:
             stack_g_prime.append(prereq)


# Step 3: Build graph G (edges p -> i if p is a prerequisite for i)
adj_G = defaultdict(list) # Stores edges p -> i where p is prereq for i
for i_node in range(1, N + 1):
     for p_node in prerequisites[i_node]:
         # Edge p_node -> i_node in G
         adj_G[p_node].append(i_node)


# Step 4: Build graph G_R (G restricted to nodes in R) and compute in_degree_R
adj_G_R = defaultdict(list) # Edges u -> v where u is prereq for v, both in R
in_degree_R = defaultdict(int) # In-degrees for nodes in R within G_R

# Initialize in-degrees for all required books to 0
for book in required_books_set:
    in_degree_R[book] = 0

# Build G_R and populate in_degree_R
# Iterate through books u that are in R
for u in required_books_set:
    # Iterate through books v for which u is a prerequisite (edges u -> v in G)
    if u in adj_G:
        # Iterate through neighbors v in the order they appear in adj_G[u]
        # This order affects the topological sort output when multiple nodes become ready.
        # No need to sort adj_G[u].
        for v in adj_G[u]:
            # If v is also in R, then edge u -> v is in G_R
            if v in required_books_set:
                adj_G_R[u].append(v)
                in_degree_R[v] += 1


# Step 5: Perform topological sort on G_R using Kahn's algorithm
Q = deque()
sorted_order = []

# Add all nodes in R with in-degree 0 to the queue
# Iterate through the required_books_set to find starting nodes.
# The iteration order of a set is not guaranteed, leading to potential variation in output.
# This is allowed by the problem ("you may print any of them").
for book in required_books_set:
    if in_degree_R[book] == 0:
        Q.append(book)

# Kahn's algorithm
while Q:
    u = Q.popleft() # Use popleft for queue behavior (FIFO)
    sorted_order.append(u)

    # Process neighbors v of u in G_R (edges u -> v)
    # Check if u exists as a key in adj_G_R before iterating
    if u in adj_G_R:
        # Process neighbors in the order they were added to adj_G_R[u]
        # This order matters for the final topological sort result when multiple neighbors' in-degrees drop to 0 simultaneously.
        # No need to sort adj_G_R[u].
        for v in adj_G_R[u]:
            in_degree_R[v] -= 1
            if in_degree_R[v] == 0:
                Q.append(v) # Append for FIFO queue

# Step 6: Print the sorted order
print(*sorted_order)