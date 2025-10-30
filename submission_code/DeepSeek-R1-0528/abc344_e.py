import sys

class Node:
    __slots__ = ('val', 'prev', 'next')
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    q = int(data[1+n])
    index = 1+n+1
    queries = []
    for i in range(q):
        t = data[index]; index += 1
        if t == '1':
            x = int(data[index]); y = int(data[index+1]); index += 2
            queries.append(('1', x, y))
        else:
            x = int(data[index]); index += 1
            queries.append(('2', x))
            
    head = Node(A[0])
    node_map = {}
    node_map[A[0]] = head
    current = head
    for i in range(1, n):
        new_node = Node(A[i])
        current.next = new_node
        new_node.prev = current
        current = new_node
        node_map[A[i]] = new_node
        
    for query in queries:
        if query[0] == '1':
            x, y = query[1], query[2]
            node_x = node_map[x]
            new_node = Node(y)
            nxt = node_x.next
            node_x.next = new_node
            new_node.prev = node_x
            new_node.next = nxt
            if nxt is not None:
                nxt.prev = new_node
            node_map[y] = new_node
        else:
            x = query[1]
            node_x = node_map[x]
            prev_node = node_x.prev
            next_node = node_x.next
            if prev_node is None:
                head = next_node
            else:
                prev_node.next = next_node
            if next_node is not None:
                next_node.prev = prev_node
            del node_map[x]
            
    result = []
    cur = head
    while cur is not None:
        result.append(str(cur.val))
        cur = cur.next
    print(" ".join(result))

if __name__ == "__main__":
    main()