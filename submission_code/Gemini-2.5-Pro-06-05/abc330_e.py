import sys
from collections import Counter

# Set a higher recursion limit for deep recursive calls in the segment tree.
# This is a good practice in competitive programming for Python, although the default
# limit is likely sufficient for this problem's constraints.
sys.setrecursionlimit(2 * 10**5 + 5)

def main():
    """
    Reads input, solves the problem using a segment tree, and prints the output.
    """
    # Use fast I/O
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    
    # The MEX is always <= N, so we build a segment tree on the range [0, N].
    RANGE_SIZE = N + 1
    tree = [0] * (4 * RANGE_SIZE)
    
    # Use a Counter to track frequencies of all numbers.
    counts = Counter(A)
    
    # --- Segment Tree Helper Functions (defined inside main to act as closures) ---
    
    def build(v, tl, tr):
        """Builds the segment tree. A leaf is 1 if the number is present, 0 otherwise."""
        if tl == tr:
            if counts[tl] > 0:
                tree[v] = 1
        else:
            tm = (tl + tr) // 2
            build(2 * v, tl, tm)
            build(2 * v + 1, tm + 1, tr)
            tree[v] = tree[2 * v] + tree[2 * v + 1]

    def update(v, tl, tr, pos, val):
        """Updates the leaf at `pos` to `val` and propagates changes up."""
        if tl == tr:
            tree[v] = val
            return
        tm = (tl + tr) // 2
        if pos <= tm:
            update(2 * v, tl, tm, pos, val)
        else:
            update(2 * v + 1, tm + 1, tr, pos, val)
        tree[v] = tree[2 * v] + tree[2 * v + 1]

    def find_first_zero(v, tl, tr):
        """Finds the smallest index k with leaf value 0 by walking the tree."""
        if tl == tr:
            return tl
        tm = (tl + tr) // 2
        # If the left child's range is not full, the first zero must be there.
        if tree[2 * v] < (tm - tl + 1):
            return find_first_zero(2 * v, tl, tm)
        else:
            # Otherwise, the left child is full, so the first zero is in the right child.
            return find_first_zero(2 * v + 1, tm + 1, tr)

    # --- Initialization ---
    build(1, 0, N)

    # --- Process Queries ---
    for _ in range(Q):
        i, x = map(int, input().split())
        i -= 1  # Convert to 0-based index

        old_val = A[i]
        new_val = x
        
        if old_val == new_val:
            print(find_first_zero(1, 0, N))
            continue

        A[i] = new_val

        # Update for the old value
        counts[old_val] -= 1
        if counts[old_val] == 0:
            if 0 <= old_val <= N:
                update(1, 0, N, old_val, 0)

        # Update for the new value
        # Check the count *before* incrementing
        if counts[new_val] == 0:
            if 0 <= new_val <= N:
                update(1, 0, N, new_val, 1)
        counts[new_val] += 1
        
        print(find_first_zero(1, 0, N))

if __name__ == "__main__":
    main()