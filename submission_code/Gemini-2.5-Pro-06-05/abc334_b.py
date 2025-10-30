# YOUR CODE HERE
def solve():
    """
    Reads input, calculates the number of Christmas trees in the given range, and prints the result.
    """
    A, M, L, R = map(int, input().split())

    # The problem is to count the number of trees in the interval [L, R].
    # A tree is at position x if x = A + k*M for some integer k.
    # This is equivalent to all tree positions x satisfying x % M == A % M.

    # First, let's find the position of the first tree that is at or after L.
    # Let this be `first_tree`.
    # `first_tree` must be >= L and `first_tree % M == A % M`.
    
    # The remainder of A when divided by M.
    # Python's % operator with a positive divisor M gives a result in [0, M-1].
    A_rem = A % M
    
    # The remainder of L when divided by M.
    L_rem = L % M
    
    # We need to add an offset to L to get to the first valid tree position.
    # The smallest non-negative offset to add to L to make its remainder A_rem is:
    # offset = (A_rem - L_rem + M) % M
    # This formula works because if L_rem <= A_rem, it gives A_rem - L_rem.
    # If L_rem > A_rem, it gives A_rem - L_rem + M, which is the correct positive offset.
    offset_to_first = (A_rem - L_rem + M) % M
    first_tree = L + offset_to_first

    # Next, let's find the position of the last tree that is at or before R.
    # Let this be `last_tree`.
    # `last_tree` must be <= R and `last_tree % M == A % M`.
    R_rem = R % M
    
    # We need to subtract an offset from R to get to the last valid tree position.
    # The smallest non-negative offset to subtract from R to make its remainder A_rem is:
    # offset = (R_rem - A_rem + M) % M
    offset_from_last = (R_rem - A_rem + M) % M
    last_tree = R - offset_from_last

    # If the calculated first tree is already past the last tree, it means no trees are in the range.
    if first_tree > last_tree:
        print(0)
    else:
        # The trees form an arithmetic progression: first_tree, first_tree + M, ..., last_tree.
        # The number of terms in such a progression is (last - first) / step + 1.
        # We use integer division `//`.
        count = (last_tree - first_tree) // M + 1
        print(count)

solve()