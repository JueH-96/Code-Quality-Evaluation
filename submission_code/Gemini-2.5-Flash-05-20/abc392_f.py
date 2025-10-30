import sys
import random

# Increase recursion limit for potentially deep trees.
# For N = 5 * 10^5, a depth of N is possible in worst-case (though average is log N).
# Set it to N + a small buffer to be safe.
sys.setrecursionlimit(5 * 10**5 + 100) 

class Node:
    def __init__(self, key):
        self.key = key
        self.priority = random.random() # Random priority for treap heap property
        self.left = None
        self.right = None
        self.size = 1 # Number of nodes in subtree rooted at this node

    def get_size(self, node):
        """Helper to get size of a node, handles None."""
        return node.size if node else 0

    def update_size(self):
        """Updates the size of the current node based on its children."""
        self.size = 1 + self.get_size(self.left) + self.get_size(self.right)

# Splits treap 't' into two treaps 'L' and 'R'
# 'L' contains the first 'k' elements (by rank)
# 'R' contains the remaining elements
def split(t, k):
    if not t:
        return None, None
    
    left_size = t.get_size(t.left)
    if k <= left_size:
        # The k-th element (0-indexed) is in the left subtree, or is not in this subtree (if k=0 and left_size=0, etc.)
        # So, the current node 't' and its right subtree go to 'R'.
        # The left subtree must be split to find the first 'k' elements.
        L, t.left = split(t.left, k)
        t.update_size() # Recalculate size after child modification
        return L, t
    else:
        # The k-th element (0-indexed) is the current node 't' itself (if k == left_size + 1), or in the right subtree.
        # So, the current node 't' and its left subtree go to 'L'.
        # The right subtree must be split for the remaining elements (k - (left_size + 1)).
        t.right, R = split(t.right, k - left_size - 1)
        t.update_size() # Recalculate size after child modification
        return t, R

# Merges two treaps 'L' and 'R'
# It's assumed that all elements in 'L' should logically come before all elements in 'R'
# The merge operation respects the heap property based on priorities.
def merge(L, R):
    if not L: return R
    if not R: return L

    if L.priority > R.priority:
        # L has higher priority, so it becomes the root.
        # Its right child will be the result of merging L's current right child with R.
        L.right = merge(L.right, R)
        L.update_size()
        return L
    else:
        # R has higher priority, so it becomes the root.
        # Its left child will be the result of merging L with R's current left child.
        R.left = merge(L, R.left)
        R.update_size()
        return R

# Inserts 'key' into treap 'root' at 'rank'-th position (0-indexed)
def insert_at_rank(root, rank, key):
    new_node = Node(key)
    if not root:
        return new_node
    
    # Split the existing treap into two parts: 'L' (elements before rank), and 'R' (elements at/after rank)
    L, R = split(root, rank)
    
    # Merge 'L', then the 'new_node', then 'R'. This places 'new_node' at the specified rank.
    return merge(merge(L, new_node), R)

# Performs an in-order traversal to collect elements in their sequence order
def get_elements_inorder(t, result_list):
    if not t:
        return
    get_elements_inorder(t.left, result_list)
    result_list.append(str(t.key)) # Convert to string for efficient joining later
    get_elements_inorder(t.right, result_list)

def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    root = None # Initialize an empty treap
    
    # Perform N insertions
    for i in range(1, N + 1):
        # P_i is 1-indexed, convert to 0-indexed rank for operations
        rank = P[i-1] - 1
        root = insert_at_rank(root, rank, i)

    final_A_elements = []
    get_elements_inorder(root, final_A_elements)
    
    # Print the final array elements separated by spaces
    sys.stdout.write(" ".join(final_A_elements) + "
")

# Run the solution
solve()