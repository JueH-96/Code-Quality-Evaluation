def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = list(map(int, input_data[1:]))

    # We are given N ≥ 4 and a strictly increasing sequence X.
    # An operation on a consecutive quadruple (in ascending order) X[i], X[i+1], X[i+2], X[i+3]
    # (0-based indexing here) will "reflect" the middle two about the midpoint of the outer two
    # if X[i+1] + X[i+2] > X[i] + X[i+3].
    #
    # After each reflection:
    #   X[i+1] becomes (X[i] + X[i+3] - X[i+1]),
    #   X[i+2] becomes (X[i] + X[i+3] - X[i+2]).
    # Then we re-sort just that local quadruple to maintain ascending order in that region.
    #
    # We repeat (in any order, as many times as we like) until no further improvement is possible.
    # The problem states that under these operations, the distinctness of coordinates is always preserved.
    # We want the minimum possible sum of final coordinates.
    #
    # A key observation is that in the "final" sorted arrangement y, no consecutive quadruple
    # can satisfy y[i+1] + y[i+2] > y[i] + y[i+3], or else another reflection would reduce the sum.
    #
    # One way to implement a solution (which is common in editorial approaches to similar problems)
    # is to iteratively "fix" local violations (the same idea as a bubble‐like method).  However,
    # naively doing repeated full scans could be O(N^2).  For N up to 2e5, that can be too large.
    #
    # A more efficient practical approach is to maintain a queue (or deque) of indices where
    # a violation *might* exist.  When we fix an index i (i.e. perform a reflection on X[i+1], X[i+2]),
    # we only need to re-check i-1, i, i+1 (those that overlap in quadruples) because other distant blocks
    # are unaffected.  In practice, this converges quickly, since each reflection moves the middle pair
    # in a way that tends not to reintroduce endless oscillations.
    #
    # Although in the worst case one could imagine many local flips, the official problem constraints
    # and the geometry of the reflections ensure it terminates in reasonable time if done carefully.
    #
    # We'll implement exactly that "queue of indices" approach:
    #   - We start with all i in [0..N-4] in a queue.
    #   - While the queue is not empty, pop an index i:
    #        if there's a violation at i (X[i+1]+X[i+2] > X[i]+X[i+3]),
    #            do the reflection,
    #            reorder X[i+1], X[i+2] if needed (small < large),
    #            then push i-1, i, i+1 into the queue (if in range).
    #     Mark indices "in progress" or visited so we don't push them repeatedly without changes.
    #
    # At the end, X is sorted overall except possibly for local reorderings we do.  But each reflection
    # only changes X[i+1], X[i+2], which remain bracketed by X[i] and X[i+3]; so we can keep the global
    # array in sorted order except for the middle pair, which we reorder locally.  This ensures X stays
    # sorted, so "consecutive quadruple" remains X[i], X[i+1], X[i+2], X[i+3] in that order.
    #
    # Finally we output sum(X).

    # Edge case: if N < 4, no operation is possible.  But by the problem statement, N ≥ 4.

    from collections import deque

    # We'll work in-place on X.  X is already sorted, so we can treat X[i], X[i+1], X[i+2], X[i+3]
    # as a consecutive quadruple in ascending order.

    def check_and_reflect(i):
        """
        Check the quadruple X[i], X[i+1], X[i+2], X[i+3].
        If there's a violation (X[i+1] + X[i+2] > X[i] + X[i+3]), reflect.
        Return True if a reflection happened, False otherwise.
        """
        if X[i+1] + X[i+2] <= X[i] + X[i+3]:
            return False
        # Perform reflection
        # new_middle1 = (X[i] + X[i+3]) - X[i+1]
        # new_middle2 = (X[i] + X[i+3]) - X[i+2]
        m1 = X[i] + X[i+3] - X[i+1]
        m2 = X[i] + X[i+3] - X[i+2]
        if m1 > m2:
            m1, m2 = m2, m1
        X[i+1], X[i+2] = m1, m2
        return True

    n = N
    if n == 4:
        # Just one possible quadruple to check once or not.
        # This trivial case can be handled by the general code anyway, but let's short-circuit.
        check_and_reflect(0)
    else:
        # General case
        q = deque(range(n - 3))
        in_queue = [False]*(n-3)  # track if index is in queue to avoid duplicates

        for i in range(n - 3):
            in_queue[i] = True

        while q:
            i = q.popleft()
            in_queue[i] = False
            if check_and_reflect(i):
                # If we did reflect, we need to re-check neighbors i-1, i, i+1
                # because it might create or fix adjacent violations.
                if i - 1 >= 0 and not in_queue[i-1]:
                    in_queue[i-1] = True
                    q.append(i-1)
                if not in_queue[i] and i < n-3:
                    in_queue[i] = True
                    q.append(i)
                if i + 1 < n-3 and not in_queue[i+1]:
                    in_queue[i+1] = True
                    q.append(i+1)

    print(sum(X))

# Do not forget to call main()!
if __name__ == "__main__":
    main()