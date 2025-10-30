import sys

class Node:
    __slots__ = ('val', 'prev', 'next')
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    n = int(data[0].strip())
    A = list(map(int, data[1].split()))
    q = int(data[2].strip())
    queries = data[3:3+q]
    
    mapping = {}
    head = None
    tail = None
    
    if n > 0:
        head = Node(A[0])
        mapping[A[0]] = head
        current = head
        for i in range(1, n):
            num = A[i]
            new_node = Node(num)
            mapping[num] = new_node
            current.next = new_node
            new_node.prev = current
            current = new_node
        tail = current
        
    for i in range(q):
        parts = queries[i].split()
        if parts[0] == '1':
            x = int(parts[1])
            y = int(parts[2])
            node_x = mapping[x]
            new_node = Node(y)
            mapping[y] = new_node
            
            next_node = node_x.next
            node_x.next = new_node
            new_node.prev = node_x
            new_node.next = next_node
            
            if next_node is not None:
                next_node.prev = new_node
            else:
                tail = new_node
                
        else:
            x = int(parts[1])
            node = mapping[x]
            prev_node = node.prev
            next_node = node.next
            
            if prev_node is not None:
                prev_node.next = next_node
            else:
                head = next_node
                
            if next_node is not None:
                next_node.prev = prev_node
            else:
                tail = prev_node
                
            del mapping[x]
            
    result = []
    current = head
    while current is not None:
        result.append(str(current.val))
        current = current.next
        
    print(" ".join(result))

if __name__ == "__main__":
    main()