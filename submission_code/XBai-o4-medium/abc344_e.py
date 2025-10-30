import sys

class Node:
    __slots__ = ('val', 'prev', 'next')
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + N]))
    ptr += N
    
    nodes = {}
    for val in A:
        nodes[val] = Node(val)
    
    head = nodes[A[0]]
    tail = nodes[A[-1]]
    
    for i in range(N - 1):
        prev_val = A[i]
        curr_val = A[i + 1]
        nodes[prev_val].next = nodes[curr_val]
        nodes[curr_val].prev = nodes[prev_val]
    
    Q = int(input[ptr])
    ptr += 1
    
    for _ in range(Q):
        type_query = int(input[ptr])
        if type_query == 1:
            x = int(input[ptr + 1])
            y = int(input[ptr + 2])
            ptr += 3
            x_node = nodes[x]
            new_node = Node(y)
            new_node.next = x_node.next
            new_node.prev = x_node
            x_node.next = new_node
            if new_node.next is not None:
                new_node.next.prev = new_node
            else:
                tail = new_node
            nodes[y] = new_node
        else:
            x = int(input[ptr + 1])
            ptr += 2
            x_node = nodes[x]
            prev_node = x_node.prev
            next_node = x_node.next
            if prev_node:
                prev_node.next = next_node
            else:
                head = next_node
            if next_node:
                next_node.prev = prev_node
            else:
                tail = prev_node
            del nodes[x]
    
    result = []
    current = head
    while current is not None:
        result.append(str(current.val))
        current = current.next
    print(' '.join(result))

if __name__ == '__main__':
    main()