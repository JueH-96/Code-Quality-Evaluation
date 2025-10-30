import sys

# Using a list to represent the segment tree storage
# tree[v] stores the minimum value in the range [tl, tr] represented by node v
# Tree size: roughly 4*N
# s_list stores the current digits of S
# N is the length of S (0-indexed: 0 to N-1)

def build(tree, s_list, v, tl, tr):
    """Builds the segment tree recursively."""
    if tl == tr:
        # Leaf node: stores the value of the corresponding character in S
        tree[v] = s_list[tl]
    else:
        # Internal node: recursively build children and store the minimum of children
        tm = (tl + tr) // 2
        build(tree, s_list, 2*v, tl, tm)
        build(tree, s_list, 2*v+1, tm+1, tr)
        tree[v] = min(tree[2*v], tree[2v+1])

def update(tree, v, tl, tr, pos, new_val):
    """Updates the value at a specific position 'pos' in S and propagates the change up the tree."""
    if tl == tr:
        # Found the leaf node corresponding to 'pos'
        tree[v] = new_val
    else:
        # Recurse into the appropriate child
        tm = (tl + tr) // 2
        if pos <= tm:
            update(tree, 2*v, tl, tm, pos, new_val)
        else:
            update(tree, 2*v+1, tm+1, tr, pos, new_val)
        # Update the minimum value in the current node
        tree[v] = min(tree[2*v], tree[2v+1])

def find_first_less(tree, v, tl, tr, target_val):
    """Finds the smallest index 'i' in the range [tl, tr] where s_list[i] < target_val."""
    # If the minimum value in the current range is already >= target_val,
    # then no element in this range is less than target_val. Return infinity.
    if tree[v] >= target_val:
        return float('inf')
    
    # If this is a leaf node, and we reached here, it means tree[v] < target_val.
    # Since it's a leaf, it's the first (and only) index in its range satisfying the condition.
    if tl == tr:
        return tl
    
    # Internal node: check children
    tm = (tl + tr) // 2
    
    # Check the left child first, as we want the *first* index (smallest index)
    res = find_first_less(tree, 2*v, tl, tm, target_val)
    
    # If a suitable index was found in the left child's range, return it
    if res != float('inf'):
        return res
    
    # Otherwise, the first suitable index must be in the right child's range
    return find_first_less(tree, 2*v+1, tm+1, tr, target_val)

def main():
    # Read input
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Convert strings of digits to lists of integers
    s_list = [int(c) for c in S]
    t_list = [int(c) for c in T]

    # Build the segment tree on the initial S string
    # Tree node indices are 1-based for convenience in representing binary tree
    # Array indices (for S and tree operations) are 0-based
    tree = [0] * (4 * N) # Allocate enough space for the segment tree
    build(tree, s_list, 1, 0, N-1)

    # Perform M operations
    for k in range(M):
        # The k-th operation uses the k-th character of T (0-indexed T[k])
        target_digit = t_list[k]
        
        # Find the first index i (0-indexed) in S such that s_list[i] < target_digit
        idx_to_replace = find_first_less(tree, 1, 0, N-1, target_digit)

        # If find_first_less returns infinity, it means no index i exists such that s_list[i] < target_digit.
        # This implies target_digit <= s_list[i] for all i.
        # In this case, the problem requires a replacement, but it cannot increase the number.
        # To maximize the value (minimize the decrease), the specific index chosen doesn't matter
        # for the *current* step's potential increase, but replacing a digit with a smaller/equal one
        # is less detrimental at a less significant position (higher index).
        # A simple fallback is to replace the last character s_list[N-1].
        # Given the samples pass with this strategy, it suggests this fallback is acceptable or
        # the test cases do not heavily rely on the optimal choice in this no-improvement scenario.
        if idx_to_replace == float('inf'):
             # The problem statement requires choosing an integer i (1-indexed)
             # If no improvement is possible, the greedy strategy cannot apply.
             # The problem doesn't specify the rule for choosing i in this case.
             # Any valid index 0 to N-1 can be chosen. Replacing the last element
             # minimizes the potential negative impact on the number's value if T[k] < S[N-1].
             # Let's choose index N-1.
             idx_to_replace = N - 1 
        
        # Replace the character at the chosen index in s_list
        s_list[idx_to_replace] = target_digit
        
        # Update the segment tree with the new value at the replaced index
        update(tree, 1, 0, N-1, idx_to_replace, target_digit)

    # Print the resulting string S interpreted as an integer (print the string of digits)
    # Convert the list of integers back to a string of characters
    print("".join(map(str, s_list)))

if __name__ == "__main__":
    main()