import sys

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

def insert_after(node, new_node):
    new_node.next = node.next
    new_node.prev = node
    node.next.prev = new_node
    node.next = new_node

def remove_node(node):
    node.prev.next = node.next
    node.next.prev = node.prev
    node.next = None
    node.prev = None

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    initial_sequence = []
    for _ in range(N):
        initial_sequence.append(int(data[ptr]))
        ptr += 1
    # Initialize linked list with sentinel nodes
    head = Node(None)
    tail = Node(None)
    head.next = tail
    tail.prev = head
    nodes = {}
    # Insert initial sequence
    for val in initial_sequence:
        node = Node(val)
        insert_after(head, node)
        nodes[val] = node
    Q = int(data[ptr])
    ptr += 1
    for _ in range(Q):
        query_type = int(data[ptr])
        ptr += 1
        if query_type == 1:
            x = int(data[ptr])
            ptr += 1
            y = int(data[ptr])
            ptr += 1
            x_node = nodes[x]
            new_node = Node(y)
            insert_after(x_node, new_node)
            nodes[y] = new_node
        elif query_type == 2:
            x = int(data[ptr])
            ptr += 1
            x_node = nodes[x]
            remove_node(x_node)
            del nodes[x]
    # Collect the final sequence
    current = head.next
    result = []
    while current != tail:
        result.append(str(current.value))
        current = current.next
    print(' '.join(result))

if __name__ == '__main__':
    main()