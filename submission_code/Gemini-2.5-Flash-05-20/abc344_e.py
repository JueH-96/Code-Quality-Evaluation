import sys

# Define the Node class for the doubly linked list
class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

# Define the DoublyLinkedList class
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, value):
        """Appends a new node with the given value to the end of the list."""
        new_node = Node(value)
        if not self.head: # List is empty
            self.head = new_node
            self.tail = new_node
        else: # List is not empty
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        return new_node # Return the node object so it can be stored in the dictionary

    def insert_after(self, existing_node, new_value):
        """
        Inserts a new node with new_value immediately after existing_node.
        Assumes existing_node is a valid node object already in the list.
        """
        new_node = Node(new_value)
        
        new_node.prev = existing_node
        new_node.next = existing_node.next
        
        if existing_node.next:
            # If existing_node is not the tail, update the next node's prev pointer
            existing_node.next.prev = new_node
        else:
            # If existing_node is the tail, the new node becomes the new tail
            self.tail = new_node
            
        existing_node.next = new_node
        return new_node # Return the new node object for dictionary mapping

    def remove(self, node_to_remove):
        """
        Removes the given node from the list.
        Assumes node_to_remove is a valid node object in the list.
        """
        if node_to_remove.prev:
            # If node_to_remove is not the head, update the previous node's next pointer
            node_to_remove.prev.next = node_to_remove.next
        else:
            # If node_to_remove is the head, update the head of the list
            self.head = node_to_remove.next
            
        if node_to_remove.next:
            # If node_to_remove is not the tail, update the next node's prev pointer
            node_to_remove.next.prev = node_to_remove.prev
        else:
            # If node_to_remove is the tail, update the tail of the list
            self.tail = node_to_remove.prev
            
        # Optional: Clear pointers of the removed node to aid garbage collection
        node_to_remove.prev = None
        node_to_remove.next = None

    def to_list(self):
        """Converts the linked list into a Python list of values."""
        result = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next
        return result

def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())
    # Read the initial sequence A from standard input
    A = list(map(int, sys.stdin.readline().split()))
    # Read Q (number of queries) from standard input
    Q = int(sys.stdin.readline())

    # Initialize the DoublyLinkedList
    dll = DoublyLinkedList()
    # Initialize the dictionary to map values to their corresponding Node objects
    value_to_node = {} 

    # Populate the doubly linked list and the hash map with initial elements
    for val in A:
        node = dll.append(val)
        value_to_node[val] = node

    # Process each query
    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        q_type = query[0]

        if q_type == 1:  # Type 1 query: Insert y immediately after x
            x, y = query[1], query[2]
            # Get the Node object for x using the hash map
            x_node = value_to_node[x]
            # Insert y as a new node after x_node
            new_y_node = dll.insert_after(x_node, y)
            # Add the new y value and its node to the hash map
            value_to_node[y] = new_y_node
        elif q_type == 2:  # Type 2 query: Remove x
            x = query[1]
            # Get the Node object for x using the hash map
            x_node = value_to_node[x]
            # Remove x_node from the doubly linked list
            dll.remove(x_node)
            # Remove x from the hash map
            del value_to_node[x]

    # After all queries are processed, convert the linked list to a Python list
    final_sequence = dll.to_list()
    # Print the final sequence elements separated by spaces, followed by a newline
    sys.stdout.write(" ".join(map(str, final_sequence)) + "
")

# Call the solve function to run the program
solve()