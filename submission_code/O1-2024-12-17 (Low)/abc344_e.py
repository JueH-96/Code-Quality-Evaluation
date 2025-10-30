import sys
sys.setrecursionlimit(10**7)

class Node:
    __slots__ = ('val', 'left', 'right')
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def main():
    input_data = sys.stdin.read().strip().split()
    # Parse the first line: N
    N = int(input_data[0])
    
    # Read the initial array A
    idx = 1
    A = list(map(int, input_data[idx:idx+N]))
    idx += N

    # Build a doubly linked list out of A
    head = Node(A[0])
    node_map = {A[0]: head}
    
    prev = head
    for value in A[1:]:
        cur = Node(value)
        cur.left = prev
        prev.right = cur
        node_map[value] = cur
        prev = cur
    
    # We'll keep track of the tail for convenience
    tail = prev

    # Number of queries
    Q = int(input_data[idx])
    idx += 1

    for _ in range(Q):
        t = input_data[idx]
        idx += 1

        if t == '1':
            # format: 1 x y -> Insert y after x
            x = int(input_data[idx])
            y = int(input_data[idx+1])
            idx += 2
            
            # Find the node for x
            node_x = node_map[x]
            # Create a new node for y
            new_node = Node(y)
            node_map[y] = new_node
            
            # Insert after node_x
            right_node = node_x.right
            
            # Link node_x -> new_node
            node_x.right = new_node
            new_node.left = node_x
            
            # Then link new_node -> right_node
            new_node.right = right_node
            if right_node is not None:
                right_node.left = new_node
            else:
                # new_node is now the tail
                tail = new_node

        else:
            # format: 2 x -> Remove x
            x = int(input_data[idx])
            idx += 1
            node_x = node_map[x]
            
            left_node = node_x.left
            right_node = node_x.right
            
            # If node_x is the head
            if left_node is None:
                # node_x was head
                head = right_node
                if right_node is not None:
                    right_node.left = None
            else:
                left_node.right = right_node
            
            # If node_x is the tail
            if right_node is None:
                # node_x was tail
                tail = left_node
                if left_node is not None:
                    left_node.right = None
            else:
                right_node.left = left_node
            
            # Remove from map
            del node_map[x]

    # Now we print the elements from head to tail
    output = []
    cur = head
    while cur is not None:
        output.append(str(cur.val))
        cur = cur.right

    print(" ".join(output))


# DO NOT FORGET TO CALL main()
if __name__ == "__main__":
    main()