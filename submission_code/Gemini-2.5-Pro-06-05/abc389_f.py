import sys
import random

# A solution using a Treap (a randomized Balanced Binary Search Tree).
# This allows for efficient splitting, merging, and lazy updates on the set of query ratings.

# Set a higher recursion limit for deep treaps, which can occur with Q nodes.
sys.setrecursionlimit(3 * 10**5 + 5)

class Node:
    """A node in the Treap."""
    def __init__(self, key, query_indices):
        self.key = key
        self.priority = random.random()
        self.query_indices = query_indices
        self.lazy = 0
        self.left = None
        self.right = None

def push(node):
    """Propagates lazy updates down to children."""
    if not node or node.lazy == 0:
        return
    node.key += node.lazy
    if node.left:
        node.left.lazy += node.lazy
    if node.right:
        node.right.lazy += node.lazy
    node.lazy = 0

def split(node, key):
    """Splits treap `node` into two: keys < `key` and keys >= `key`."""
    if not node:
        return None, None
    push(node)
    if node.key < key:
        left_sub, right_sub = split(node.right, key)
        node.right = left_sub
        return node, right_sub
    else:
        left_sub, right_sub = split(node.left, key)
        node.left = right_sub
        return left_sub, node

def merge(left, right):
    """Merges two treaps. Assumes max(left.keys) < min(right.keys)."""
    if not left: return right
    if not right: return left
    
    push(left)
    push(right)
    
    if left.priority > right.priority:
        left.right = merge(left.right, right)
        return left
    else:
        right.left = merge(left, right.left)
        return right

def insert(root, new_node):
    """Inserts a new node, handling key collisions by merging query indices."""
    if not root:
        return new_node
    
    push(root)
    
    if new_node.key == root.key:
        root.query_indices.extend(new_node.query_indices)
        return root

    if new_node.priority > root.priority:
        new_node.left, new_node.right = split(root, new_node.key)
        return new_node
    
    if new_node.key < root.key:
        root.left = insert(root.left, new_node)
    else:
        root.right = insert(root.right, new_node)
    return root

def merge_colliding(t1, t2):
    """Merges two treaps that may have the same keys."""
    if not t1: return t2
    if not t2: return t1
    
    push(t1)
    push(t2)
    
    if t1.priority < t2.priority:
        t1, t2 = t2, t1
    
    left, right = split(t2, t1.key)
    # The split gives left part (< t1.key) and right part (>= t1.key)
    # The right part may contain a node with key equal to t1.key.
    
    mid, right = split(right, t1.key + 1)
    if mid: # key collision
        t1.query_indices.extend(mid.query_indices)

    t1.left = merge_colliding(t1.left, left)
    t1.right = merge_colliding(t1.right, right)
    return t1


def to_list(node, result_array):
    """In-order traversal to populate the final answer array."""
    if not node:
        return
    push(node)
    to_list(node.left, result_array)
    for q_idx in node.query_indices:
        result_array[q_idx] = node.key
    to_list(node.right, result_array)

def main():
    # Fast I/O
    lines = sys.stdin.readlines()
    line_idx = 0

    N = int(lines[line_idx])
    line_idx += 1
    contests = []
    for _ in range(N):
        contests.append(tuple(map(int, lines[line_idx].split())))
        line_idx += 1
    
    Q = int(lines[line_idx])
    line_idx += 1
    
    queries_map = {}
    for i in range(Q):
        x = int(lines[line_idx])
        line_idx += 1
        if x not in queries_map:
            queries_map[x] = []
        queries_map[x].append(i)
        
    root = None
    sorted_keys = sorted(queries_map.keys())
    
    # Build the treap from initial queries
    for key in sorted_keys:
        root = insert(root, Node(key, queries_map[key]))

    for L, R in contests:
        # Split treap into three parts: < L, [L, R], > R
        t_left, t_ge_L = split(root, L)
        t_mid, t_right = split(t_ge_L, R + 1)
        
        if t_mid:
            t_mid.lazy += 1
        
        # Merge back parts. `t_left` and `t_mid` are guaranteed not to have key collisions after `t_mid` is incremented.
        root = merge(t_left, t_mid)
        # `root` and `t_right` might have collisions at R+1.
        root = merge_colliding(root, t_right)

    ans = [0] * Q
    to_list(root, ans)
    
    # Using sys.stdout.write for speed
    output = []
    for val in ans:
        output.append(str(val))
    sys.stdout.write('
'.join(output) + '
')

if __name__ == "__main__":
    main()