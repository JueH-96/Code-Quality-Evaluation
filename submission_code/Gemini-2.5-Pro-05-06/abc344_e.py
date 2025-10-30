import sys

# Node class for the doubly linked list
class Node:
    __slots__ = 'value', 'prev', 'next' # Optimize memory
    def __init__(self, value):
        self.value = value
        self.prev = None # Will be set by linked list logic
        self.next = None # Will be set by linked list logic

def solve():
    # N is read but not directly used later other than to know the size of A_initial.
    # A_initial itself determines the loop count.
    _N = int(sys.stdin.readline()) 
    A_initial = list(map(int, sys.stdin.readline().split()))
    Q_count = int(sys.stdin.readline())

    # Initialize sentinels
    # These dummy nodes simplify list operations by ensuring every real node
    # has non-None prev and next (pointing to a sentinel if at an end).
    head_sentinel = Node(None) # Value for sentinels can be anything not in data range
    tail_sentinel = Node(None)
    head_sentinel.next = tail_sentinel
    tail_sentinel.prev = head_sentinel

    # Map values to their Node objects for O(1) average time access
    value_to_node_map = {} 

    # Populate initial list
    # Structure: head_sentinel <-> A_initial[0] <-> A_initial[1] <-> ... <-> tail_sentinel
    current_tail_ptr = head_sentinel # Tracks the node after which the next new node will be inserted
    for val in A_initial:
        new_node = Node(val)
        value_to_node_map[val] = new_node
        
        # Insert new_node after current_tail_ptr:
        # current_tail_ptr <-> new_node <-> current_tail_ptr.next (which is tail_sentinel initially, or for the last element)
        new_node.prev = current_tail_ptr
        new_node.next = current_tail_ptr.next 
        
        current_tail_ptr.next.prev = new_node # Update backward link of node that was after current_tail_ptr
        current_tail_ptr.next = new_node      # Update forward link of current_tail_ptr
        
        current_tail_ptr = new_node # The new_node is now the effective tail of the constructed part
    
    # Process queries
    for _ in range(Q_count):
        query_line_parts = sys.stdin.readline().split()
        query_type = int(query_line_parts[0])

        if query_type == 1: # Insert y_val immediately after x_val
            x_val = int(query_line_parts[1])
            y_val = int(query_line_parts[2])
            
            node_x = value_to_node_map[x_val]
            new_node_y = Node(y_val)
            value_to_node_map[y_val] = new_node_y
            
            # Links for new_node_y:
            # current structure: node_x <-> node_x.next
            # target structure:  node_x <-> new_node_y <-> node_x.next
            new_node_y.prev = node_x
            new_node_y.next = node_x.next
            
            # Update links of neighbors:
            node_x.next.prev = new_node_y # node_x.next is never None due to tail_sentinel
            node_x.next = new_node_y
            
        else: # query_type == 2: Remove x_val
            x_val = int(query_line_parts[1])
            
            node_to_remove = value_to_node_map[x_val]
            
            # Bypass node_to_remove:
            # current structure: node_to_remove.prev <-> node_to_remove <-> node_to_remove.next
            # target structure:  node_to_remove.prev <-> node_to_remove.next
            node_to_remove.prev.next = node_to_remove.next
            node_to_remove.next.prev = node_to_remove.prev
            
            del value_to_node_map[x_val]
            # Python's garbage collector will reclaim memory for node_to_remove.
            # No need to explicitly nullify its pointers (node_to_remove.prev = None etc.)
            # unless trying to break complex reference cycles, not an issue here.

    # Collect results for printing
    result_values = []
    current_node = head_sentinel.next # First actual data node
    while current_node != tail_sentinel:
        result_values.append(str(current_node.value))
        current_node = current_node.next
    
    sys.stdout.write(" ".join(result_values) + "
")

if __name__ == '__main__':
    solve()