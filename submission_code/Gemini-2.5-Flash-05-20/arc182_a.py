import sys
from collections import defaultdict

# It's crucial for Python to optimize I/O
input = sys.stdin.readline

def solve():
    N, Q = map(int, input().split())
    
    ops = []
    for _ in range(Q):
        p, v = map(int, input().split())
        ops.append((p, v))

    # Compress V values to indices
    all_V_values = {0} # S starts with 0s
    for _, v in ops:
        all_V_values.add(v)
    
    V_list = sorted(list(all_V_values))
    V_to_idx = {v: i for i, v in enumerate(V_list)}
    
    MOD = 998244353

    # Persistent Segment Tree Node
    # Each node represents a range [l, r] in the S array.
    # val_idx: The maximum value (compressed index) in this range. This value already incorporates any lazy tags.
    # lazy_val_idx: A value (compressed index) that needs to be propagated to children.
    # left_node, right_node: References to child nodes.
    
    # Node cache to ensure canonical representation and avoid redundant Node objects.
    # Key: (val_idx, lazy_val_idx, hash(left_node), hash(right_node)) -> Node object
    node_cache_map = {}

    class Node:
        # Using __slots__ for memory optimization in Python classes with many instances
        __slots__ = 'val_idx', 'lazy_val_idx', 'left_node', 'right_node', '_hash_val'

        def __init__(self, val_idx, lazy_val_idx, left_node=None, right_node=None):
            self.val_idx = val_idx
            self.lazy_val_idx = lazy_val_idx
            self.left_node = left_node
            self.right_node = right_node
            self._hash_val = None # Cache hash value for memoization

        def __hash__(self):
            if self._hash_val is None:
                # Hash is based on content (values and child hashes)
                self._hash_val = hash((self.val_idx, self.lazy_val_idx, 
                                        hash(self.left_node) if self.left_node else 0, # 0 for None
                                        hash(self.right_node) if self.right_node else 0))
            return self._hash_val

        def __eq__(self, other):
            if not isinstance(other, Node):
                return NotImplemented
            # Equality check based on content
            return (self.val_idx == other.val_idx and
                    self.lazy_val_idx == other.lazy_val_idx and
                    self.left_node == other.left_node and
                    self.right_node == other.right_node)
        
        # For debugging purposes
        # def __repr__(self):
        #     return f"Node(val={V_list[self.val_idx]}, lazy={V_list[self.lazy_val_idx]}, L={id(self.left_node) if self.left_node else 'None'}, R={id(self.right_node) if self.right_node else 'None'})"

    # Factory function to get canonical Node instances from cache
    def get_canonical_node(val_idx, lazy_val_idx, left_node, right_node):
        # Create a temporary Node instance to compute its hash key
        temp_node = Node(val_idx, lazy_val_idx, left_node, right_node)
        node_key = temp_node.__hash__() # Use __hash__ to get the unique key

        if node_key not in node_cache_map:
            node_cache_map[node_key] = temp_node # Store the actual Node object
        return node_cache_map[node_key]

    # Segment tree build operation
    def build_tree(start, end):
        if start == end: # Leaf node
            return get_canonical_node(V_to_idx[0], 0, None, None) # Value 0, no lazy, no children
        
        mid = (start + end) // 2
        left_child = build_tree(start, mid)
        right_child = build_tree(mid + 1, end)
        
        # Internal node: val_idx is max of children's val_idx, no lazy initially
        return get_canonical_node(max(left_child.val_idx, right_child.val_idx), 0, left_child, right_child)

    # Helper function to apply a lazy tag to a node, returning a new canonical node.
    # This also applies the lazy tag to the node's current value.
    def apply_lazy(node, new_lazy_val_idx):
        if node is None: # For conceptual null children, create a new leaf node with the lazy value
            return get_canonical_node(new_lazy_val_idx, new_lazy_val_idx, None, None)
        
        # Create a new node with updated val_idx and lazy_val_idx (max-update behavior)
        return get_canonical_node(
            max(node.val_idx, new_lazy_val_idx),  # Node's value is max of current and new lazy
            max(node.lazy_val_idx, new_lazy_val_idx), # Lazy tag accumulates (max-update)
            node.left_node, node.right_node
        )

    # Segment tree update operation (range max-update)
    # Returns the new root node of the updated tree segment
    def update_range(node, start, end, l, r, val_idx):
        # Base cases for recursion
        if r < start or end < l: # Range completely outside current node's range
            return node # No change to this part of the tree
        
        # If node's range is completely within update range, apply value and lazy tag
        if l <= start and end <= r:
            return apply_lazy(node, val_idx)
        
        # Non-leaf node, partially overlaps:
        # First, conceptually push current node's lazy tag to its children
        new_left_node = apply_lazy(node.left_node, node.lazy_val_idx)
        new_right_node = apply_lazy(node.right_node, node.lazy_val_idx)

        # Recurse on children with the new value
        mid = (start + end) // 2
        updated_left = update_range(new_left_node, start, mid, l, r, val_idx)
        updated_right = update_range(new_right_node, mid + 1, end, l, r, val_idx)
        
        # Construct and return the new parent node
        # Its lazy tag is reset to 0 as it's propagated.
        # Its val_idx is the max of its updated children's val_idx.
        return get_canonical_node(
            max(updated_left.val_idx, updated_right.val_idx),
            0, # Lazy tag has been propagated
            updated_left, updated_right
        )

    # Segment tree query operation (range max query)
    # Returns the maximum value (compressed index) in the queried range
    def query_range(node, start, end, l, r):
        # Base cases
        if r < start or end < l or node is None:
            return V_to_idx[0] # Return index for 0 (smallest value) if range completely outside or node is None

        # If node's range is completely within query range, its val_idx holds the max.
        # Node's val_idx already reflects its own lazy_val_idx because of `apply_lazy` in update_range.
        if l <= start and end <= r:
            return node.val_idx 

        # Partially overlaps: recurse on children
        mid = (start + end) // 2
        
        # Query results from children. Children's values implicitly incorporate parent's lazy tag when they were created by `apply_lazy`.
        res_from_children = V_to_idx[0]
        if l <= mid:
            res_from_children = max(res_from_children, query_range(node.left_node, start, mid, l, r))
        if r > mid:
            res_from_children = max(res_from_children, query_range(node.right_node, mid + 1, end, l, r))
        
        # Combine max from children and current node's lazy tag (if any)
        # Node's lazy_val_idx applies to its entire range, so it could be the max value in any sub-range.
        return max(res_from_children, node.lazy_val_idx)
    

    # DP memoization table: map Node instance (representing current tree state) to count
    dp_memo = {} 

    # Recursive DP function
    def solve_dp(op_idx, current_tree_root):
        if op_idx == Q:
            return 1 # Base case: All operations processed successfully

        # Memoization check
        state_key = hash(current_tree_root)
        if state_key in dp_memo:
            return dp_memo[state_key]

        P, V = ops[op_idx]
        V_idx = V_to_idx[V]

        current_ways = 0

        # Choice 1: Replace S_1 ... S_P with V
        # Check condition: max(S_1 ... S_P) <= V
        max_val_left_range_idx = query_range(current_tree_root, 1, N, 1, P)
        max_val_left_range = V_list[max_val_left_range_idx] # Convert back to actual value for comparison
        
        if max_val_left_range <= V:
            # If valid, apply update and recurse
            new_tree_root_left = update_range(current_tree_root, 1, N, 1, P, V_idx)
            current_ways = (current_ways + solve_dp(op_idx + 1, new_tree_root_left)) % MOD

        # Choice 2: Replace S_P ... S_N with V
        # Check condition: max(S_P ... S_N) <= V
        max_val_right_range_idx = query_range(current_tree_root, 1, N, P, N)
        max_val_right_range = V_list[max_val_right_range_idx] # Convert back to actual value for comparison

        if max_val_right_range <= V:
            # If valid, apply update and recurse
            new_tree_root_right = update_range(current_tree_root, 1, N, P, N, V_idx)
            current_ways = (current_ways + solve_dp(op_idx + 1, new_tree_root_right)) % MOD

        # Store result in memo and return
        dp_memo[state_key] = current_ways
        return current_ways

    # Set recursion limit higher for potentially deep DP calls
    sys.setrecursionlimit(max(sys.getrecursionlimit(), Q + 100)) 

    # Initialize segment tree: all S elements are 0
    initial_root = build_tree(1, N)
    
    # Run the DP
    result = solve_dp(0, initial_root)
    print(result)

# Call the main solve function
solve()