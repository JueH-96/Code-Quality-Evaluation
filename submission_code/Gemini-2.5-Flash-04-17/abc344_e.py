import sys

# Node class for Doubly Linked List
class Node:
    __slots__ = ('value', 'prev', 'next') # Optimization: use slots to save memory

    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

# Read N
N = int(sys.stdin.readline())

# Dummy head and tail nodes
head = Node(None)
tail = Node(None)
head.next = tail
tail.prev = head

# Dictionary to map values to nodes for O(1) lookup
value_to_node = {}

# Build the initial doubly linked list and populate the dictionary
initial_values = map(int, sys.stdin.readline().split())
current_last_node = head # Pointer to the last node added (initially the dummy head)

for value in initial_values:
    new_node = Node(value)

    # Link new_node after current_last_node and before tail
    new_node.prev = current_last_node
    new_node.next = tail

    current_last_node.next = new_node
    tail.prev = new_node

    value_to_node[value] = new_node
    current_last_node = new_node # Update the pointer

# Read Q
Q = int(sys.stdin.readline())

# Process queries
for _ in range(Q):
    query = list(map(int, sys.stdin.readline().split()))
    type = query[0]

    if type == 1: # Insert y after x
        x, y = query[1], query[2]
        # x is guaranteed to exist in the list
        node_x = value_to_node[x]

        # Create the new node for y
        node_y = Node(y)

        # Get the node currently after x
        node_after_x = node_x.next

        # Link node_y into the list
        node_y.prev = node_x
        node_y.next = node_after_x

        # Update neighbors' links
        node_x.next = node_y
        node_after_x.prev = node_y

        # Add y to the dictionary
        # y is guaranteed to be distinct from existing elements
        value_to_node[y] = node_y

    elif type == 2: # Remove x
        x = query[1]
        # x is guaranteed to exist in the list
        node_x = value_to_node[x]

        # Get neighbors
        node_prev = node_x.prev
        node_next = node_x.next

        # Unlink node_x by linking its neighbors together
        node_prev.next = node_next
        node_next.prev = node_prev

        # Remove x from dictionary
        del value_to_node[x]

        # Optional: Clear node_x pointers for garbage collection (not strictly necessary in Python)
        # node_x.prev = None
        # node_x.next = None

# Print the final sequence
result = []
current_node = head.next # Start from the first actual node (after the dummy head)
while current_node != tail: # Stop when we reach the dummy tail
    result.append(str(current_node.value))
    current_node = current_node.next # Move to the next node

# Print the result separated by spaces
# Using sys.stdout.write is generally faster for large outputs
sys.stdout.write(" ".join(result) + "
")