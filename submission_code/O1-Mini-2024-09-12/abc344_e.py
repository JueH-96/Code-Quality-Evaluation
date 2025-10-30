import sys
import sys
import sys

class Node:
    __slots__ = ['val', 'prev', 'next']
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx += 1
    A = list(map(int, data[idx:idx+N])); idx += N
    Q = int(data[idx]); idx += 1
    queries = data[idx:]
    
    nodes = {}
    head = Node(A[0])
    nodes[A[0]] = head
    prev = head
    for val in A[1:]:
        node = Node(val)
        node.prev = prev
        prev.next = node
        nodes[val] = node
        prev = node
    tail = prev
    
    q_idx = 0
    for _ in range(Q):
        t = queries[q_idx]
        if t == '1':
            x = int(queries[q_idx+1])
            y = int(queries[q_idx+2])
            q_idx += 3
            node_x = nodes[x]
            node_y = Node(y)
            node_y.prev = node_x
            node_y.next = node_x.next
            if node_x.next:
                node_x.next.prev = node_y
            node_x.next = node_y
            nodes[y] = node_y
            if node_x == tail:
                tail = node_y
        else:
            x = int(queries[q_idx+1])
            q_idx += 2
            node_x = nodes[x]
            if node_x.prev:
                node_x.prev.next = node_x.next
            else:
                head = node_x.next
            if node_x.next:
                node_x.next.prev = node_x.prev
            else:
                tail = node_x.prev
            del nodes[x]
    
    result = []
    curr = head
    while curr:
        result.append(str(curr.val))
        curr = curr.next
    print(' '.join(result))

if __name__ == "__main__":
    main()