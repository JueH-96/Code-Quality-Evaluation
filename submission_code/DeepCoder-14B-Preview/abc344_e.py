class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0

    n = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+n]))
    ptr += n

    # Create nodes and build the initial linked list
    nodes = [Node(a) for a in A]
    for i in range(n-1):
        nodes[i].next = nodes[i+1]
        nodes[i+1].prev = nodes[i]
    head = nodes[0] if n > 0 else None
    tail = nodes[-1] if n > 0 else None

    # Build the node_map
    node_map = {a: nodes[i] for i, a in enumerate(A)}

    q = int(input[ptr])
    ptr += 1
    for _ in range(q):
        query_type = input[ptr]
        if query_type == '1':
            # Insert operation
            x = int(input[ptr+1])
            y = int(input[ptr+2])
            ptr += 3
            x_node = node_map[x]
            new_node = Node(y)
            next_node = x_node.next
            x_node.next = new_node
            new_node.prev = x_node
            new_node.next = next_node
            if next_node is not None:
                next_node.prev = new_node
            else:
                tail = new_node
            node_map[y] = new_node
        else:
            # Delete operation
            x = int(input[ptr+1])
            ptr += 2
            x_node = node_map[x]
            prev_node = x_node.prev
            next_node = x_node.next

            if prev_node is not None:
                prev_node.next = next_node
            else:
                head = next_node

            if next_node is not None:
                next_node.prev = prev_node
            else:
                tail = prev_node

            del node_map[x]

    # Collect the result
    result = []
    current = head
    while current is not None:
        result.append(str(current.value))
        current = current.next
    print(' '.join(result))

if __name__ == "__main__":
    main()