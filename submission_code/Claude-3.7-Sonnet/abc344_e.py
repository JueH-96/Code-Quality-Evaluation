class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.value_to_node = {}  # Maps values to their nodes for O(1) lookup

    def append(self, value):
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.value_to_node[value] = new_node

    def insert_after(self, x, y):
        node_x = self.value_to_node[x]
        new_node = Node(y)
        
        new_node.next = node_x.next
        new_node.prev = node_x
        
        if node_x.next:
            node_x.next.prev = new_node
        else:
            self.tail = new_node
            
        node_x.next = new_node
        self.value_to_node[y] = new_node

    def remove(self, x):
        node_to_remove = self.value_to_node[x]
        
        if node_to_remove.prev:
            node_to_remove.prev.next = node_to_remove.next
        else:
            self.head = node_to_remove.next
            
        if node_to_remove.next:
            node_to_remove.next.prev = node_to_remove.prev
        else:
            self.tail = node_to_remove.prev
            
        del self.value_to_node[x]

    def to_list(self):
        result = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next
        return result

# Read input
N = int(input())
A = list(map(int, input().split()))
Q = int(input())

# Initialize the linked list with the initial sequence
dll = DoublyLinkedList()
for a in A:
    dll.append(a)

# Process queries
for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:  # Insert y after x
        _, x, y = query
        dll.insert_after(x, y)
    else:  # Remove x
        _, x = query
        dll.remove(x)

# Print the final sequence
print(' '.join(map(str, dll.to_list())))