import sys

# Define the Node class for our doubly linked list.
class Node:
    """A node in a doubly linked list."""
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

# Read inputs using sys.stdin.readline for better performance with large inputs.
try:
    N_str = sys.stdin.readline()
    if not N_str:
        # Handle case of empty input
        exit()
    N = int(N_str)
    A = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
except (ValueError, IndexError):
    # Handle malformed input
    exit()

# `pos` is a dictionary that maps a value to its Node object for O(1) average-time access.
pos = {}

# --- Initialization Phase ---
# Create Node objects for each initial element and store them in the `pos` map.
for val in A:
    pos[val] = Node(val)

# Link the nodes together to form the initial doubly linked list.
for i in range(N - 1):
    pos[A[i]].next = pos[A[i+1]]
    pos[A[i+1]].prev = pos[A[i]]

# --- Query Processing Phase ---
for _ in range(Q):
    query_line = sys.stdin.readline()
    if not query_line:
        break
    query = list(map(int, query_line.split()))
    query_type = query[0]

    if query_type == 1:
        # Type 1: Insert y immediately after x
        _, x, y = query
        
        node_x = pos[x]
        node_after_x = node_x.next
        
        # Create the new node for y and add it to the map.
        new_node = Node(y)
        pos[y] = new_node
        
        # Update pointers to insert the new node between node_x and node_after_x.
        node_x.next = new_node
        new_node.prev = node_x
        new_node.next = node_after_x
        
        if node_after_x is not None:
            node_after_x.prev = new_node
            
    else:  # query_type == 2
        # Type 2: Remove x
        _, x = query
        
        # Find the node to remove and its neighbors.
        node_to_remove = pos[x]
        node_before = node_to_remove.prev
        node_after = node_to_remove.next
        
        # Update neighbors' pointers to bypass the removed node.
        if node_before is not None:
            node_before.next = node_after
        if node_after is not None:
            node_after.prev = node_before
        
        # Remove the value from our position map.
        del pos[x]

# --- Output Phase ---
# The problem guarantees the list is not empty, so `pos` is not empty.
# To print the list, we first need to find its head.
# We can start from any existing node and traverse backwards.

# Get an arbitrary node from the dictionary.
start_node = next(iter(pos.values()))

# Traverse to the beginning of the list to find the head.
while start_node.prev is not None:
    start_node = start_node.prev

# Now, start_node is the head. Traverse forward and collect values for printing.
result = []
current = start_node
while current is not None:
    result.append(str(current.value))
    current = current.next

print(" ".join(result))