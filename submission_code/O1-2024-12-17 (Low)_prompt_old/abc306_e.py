def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N = int(input_data[0])
    K = int(input_data[1])
    Q = int(input_data[2])

    # The updates: (X_i, Y_i)
    XY = input_data[3:]

    # We'll maintain two heaps with lazy deletions:
    #
    #   - "large" (min-heap) holds up to K largest elements (value, index).
    #        We'll store +value in this min-heap so the smallest of these top-K
    #        is easy to pop. The sum of these values is what we need to output.
    #
    #   - "small" (max-heap) holds the other elements (value, index).
    #        We'll store -value in this max-heap so the largest of "small"
    #        is easy to compare with the smallest of "large".
    #
    # Because each element A_i can move between "small" and "large",
    # we also track sum_large (sum of values in "large").
    #
    # We'll store the current location of each index (0 for small, 1 for large)
    # in an array "where". At each update, we remove the old (value, index) from
    # whichever heap it was in (lazily) and insert the new (value, index) into
    # "small" first. Then we rebalance the heaps so that "large" has exactly K
    # elements (or all elements if K > number_of_non-removed, but K ≤ N so it fits).
    #
    # Each operation (push, pop, lazy removal, rebalance) takes O(log N),
    # which should be efficient enough given the constraints.

    import heapq

    # Heaps
    small = []  # max-heap (stores (-value, index))
    large = []  # min-heap (stores (value, index))

    # Lazy removal dictionaries: how many times a (value, index) is pending removal.
    remove_small = {}
    remove_large = {}

    # Where does each index currently live? 0 -> small, 1 -> large
    where = [0]*N  # initially all are 0, but effectively all values are 0 so it doesn't matter

    # Current array A
    A = [0]*N

    # Current size of "large" and sum of values in "large"
    size_large = 0
    sum_large = 0

    def actual_top_small():
        """Pop top of small while it's marked for removal, and return
           the top element (as (-value, idx)) if it exists, otherwise None."""
        while small and remove_small.get(small[0], 0) > 0:
            top = small[0]
            remove_small[top] -= 1
            if remove_small[top] == 0:
                del remove_small[top]
            heapq.heappop(small)
        return small[0] if small else None

    def actual_top_large():
        """Pop top of large while it's marked for removal, and return
           the top element (as (value, idx)) if it exists, otherwise None."""
        while large and remove_large.get(large[0], 0) > 0:
            top = large[0]
            remove_large[top] -= 1
            if remove_large[top] == 0:
                del remove_large[top]
            heapq.heappop(large)
        return large[0] if large else None

    def push_small(val, idx):
        # push into small (as -val)
        where[idx] = 0
        heapq.heappush(small, (-val, idx))

    def push_large(val, idx):
        # push into large (as val)
        nonlocal sum_large, size_large
        where[idx] = 1
        heapq.heappush(large, (val, idx))
        sum_large += val
        size_large += 1

    def pop_small():
        # remove the actual top of small and return (val, idx) as positive val
        top = actual_top_small()
        if top is None:
            return None
        heapq.heappop(small)
        (neg_val, idx) = top
        val = -neg_val
        return (val, idx)

    def pop_large():
        # remove the actual top of large and return (val, idx)
        nonlocal sum_large, size_large
        top = actual_top_large()
        if top is None:
            return None
        heapq.heappop(large)
        (val, idx) = top
        sum_large -= val
        size_large -= 1
        return (val, idx)

    def erase_small(val, idx):
        # lazy-remove (-(val), idx) from small
        entry = (-val, idx)
        if entry not in remove_small:
            remove_small[entry] = 0
        remove_small[entry] += 1

    def erase_large(val, idx):
        # lazy-remove ((val), idx) from large
        nonlocal sum_large, size_large
        entry = (val, idx)
        if where[idx] == 1:
            # If it is truly in large, adjust sum
            sum_large -= val
            size_large -= 1
        if entry not in remove_large:
            remove_large[entry] = 0
        remove_large[entry] += 1

    # Initially, all A_i = 0
    # We want "large" to have min(K, N) elements, but they are all zero. 
    # Instead of pushing N zeros, we can do the following:
    # push K zeros into large, and the other N-K zeros into small (if K < N).
    # That way we start from a correct state.
    init_k = min(K, N)
    for i in range(init_k):
        push_large(0, i)
    for i in range(init_k, N):
        push_small(0, i)

    # Process each update
    idx_ptr = 0
    out = []
    for _ in range(Q):
        x_i = int(XY[idx_ptr]); y_i = int(XY[idx_ptr+1])
        idx_ptr += 2
        x_i -= 1  # make 0-based

        old_val = A[x_i]
        A[x_i] = y_i

        # Erase old_val from whichever heap it was
        if where[x_i] == 1:
            # it was in large
            erase_large(old_val, x_i)
        else:
            # it was in small
            erase_small(old_val, x_i)

        # Insert new_val into small first (we'll rebalance after)
        push_small(y_i, x_i)

        # Rebalance so that large has exactly K elements
        # 1) if large has fewer than K, move from small -> large
        while size_large < K:
            top = pop_small()
            if top is None:
                break
            val, idx = top
            push_large(val, idx)

        # 2) if large has more than K, move from large -> small
        while size_large > K:
            top = pop_large()
            if top is None:
                break
            val, idx = top
            push_small(val, idx)

        # 3) Compare top of small (max) vs top of large (min),
        #    if top_small > top_large, we swap one each
        while True:
            top_s = actual_top_small()
            top_l = actual_top_large()
            if not top_s or not top_l:
                # can't compare if one is empty
                break
            val_s, idx_s = -top_s[0], top_s[1]
            val_l, idx_l = top_l[0], top_l[1]
            if val_s > val_l:
                # pop from small, pop from large
                pop_small()
                pop_large()
                # push them crosswise
                push_large(val_s, idx_s)
                push_small(val_l, idx_l)
            else:
                break

        # Now sum_large is f(A)
        out.append(str(sum_large))

    # Print results
    print("
".join(out))

# Let's call solve() to comply with the problem requirement.
def main():
    solve()

if __name__ == "__main__":
    main()