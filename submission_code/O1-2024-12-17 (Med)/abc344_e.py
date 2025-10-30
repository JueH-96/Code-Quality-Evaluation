def main():
    import sys
    data = sys.stdin.read().strip().split()
    idx = 0
    
    N = int(data[idx]); idx += 1
    initial_values = list(map(int, data[idx:idx+N]))
    idx += N
    Q = int(data[idx]); idx += 1
    
    queries = []
    for _ in range(Q):
        t = data[idx]
        idx += 1
        if t == '1':
            x = int(data[idx]); idx += 1
            y = int(data[idx]); idx += 1
            queries.append((1, x, y))
        else:  # t == '2'
            x = int(data[idx]); idx += 1
            queries.append((2, x))

    class Node:
        __slots__ = ('val','prev','next')
        def __init__(self, val):
            self.val = val
            self.prev = None
            self.next = None

    # Build initial doubly linked list and store references in a dict
    nodes = {}
    head = Node(initial_values[0])
    nodes[initial_values[0]] = head
    prev_node = head
    for v in initial_values[1:]:
        new_node = Node(v)
        nodes[v] = new_node
        prev_node.next = new_node
        new_node.prev = prev_node
        prev_node = new_node

    # Process queries
    for q in queries:
        if q[0] == 1:
            # 1 x y -> Insert y immediately after x
            _, x, y = q
            node_x = nodes[x]
            node_y = Node(y)
            nodes[y] = node_y
            
            node_y.prev = node_x
            node_y.next = node_x.next
            if node_x.next is not None:
                node_x.next.prev = node_y
            node_x.next = node_y
        else:
            # 2 x -> Remove x
            _, x = q
            node_x = nodes[x]
            p = node_x.prev
            n = node_x.next
            if p is not None:
                p.next = n
            else:
                # node_x is head
                head = n
            if n is not None:
                n.prev = p
            del nodes[x]

    # Collect final sequence
    result = []
    cur = head
    while cur is not None:
        result.append(cur.val)
        cur = cur.next

    print(" ".join(map(str, result)))


# Do not forget to call main
main()