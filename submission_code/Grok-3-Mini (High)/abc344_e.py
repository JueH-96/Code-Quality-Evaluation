import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read the initial sequence A
A = list(map(int, data[index:index + N]))
index += N

# Read Q
Q = int(data[index])
index += 1

# Define Node class for doubly linked list
class Node:
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

# Build the initial doubly linked list
# Create the first node
head_node = Node(A[0])
val_to_node = {A[0]: head_node}
current = head_node

# Link the remaining nodes
for i in range(1, N):
    new_node = Node(A[i])
    new_node.prev = current
    current.next = new_node
    val_to_node[A[i]] = new_node
    current = new_node

# Set head and tail
tail = current
head = head_node

# Process Q queries
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        # Insert y after x
        x = int(data[index])  # x is the value to insert after
        index += 1
        y = int(data[index])  # y is the value to insert
        index += 1
        x_node = val_to_node[x]
        new_node = Node(y)
        new_node.next = x_node.next
        new_node.prev = x_node
        if x_node.next is not None:
            x_node.next.prev = new_node
        else:
            tail = new_node  # Update tail if inserting after the last node
        x_node.next = new_node
        val_to_node[y] = new_node  # Add new node to dictionary
    elif query_type == 2:
        # Remove x
        x_rem = int(data[index])  # x is the value to remove
        index += 1
        x_node = val_to_node[x_rem]
        prev_node = x_node.prev
        next_node = x_node.next
        if prev_node is not None:
            prev_node.next = next_node
        else:
            head = next_node  # Update head if removing the first node
        if next_node is not None:
            next_node.prev = prev_node
        else:
            tail = prev_node  # Update tail if removing the last node
        del val_to_node[x_rem]  # Remove from dictionary

# Traverse the linked list and collect values
current = head
vals = []
while current is not None:
    vals.append(current.val)
    current = current.next

# Print the values separated by spaces
print(' '.join(map(str, vals)))