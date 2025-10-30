def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast IO helpers
    # We'll parse all inputs at once and step through them.
    # This helps avoid overhead of calling input() repeatedly.
    idx_data = 0
    def read_int():
        nonlocal idx_data
        val = int(input_data[idx_data])
        idx_data += 1
        return val

    # Read N, K, Q
    N = read_int()
    K = read_int()
    Q = read_int()

    # A[i] will hold the current value of A_i. Initially 0.
    A = [0]*N

    # We'll maintain two heaps:
    # "low"  (max-heap of the "rest"), implemented by pushing negative values
    # "high" (min-heap of the "top K")
    import heapq

    low = []
    high = []

    # freqLow[val] = how many times 'val' is conceptually in "low"
    # freqHigh[val] = how many times 'val' is conceptually in "high"
    from collections import defaultdict
    freqLow = defaultdict(int)
    freqHigh = defaultdict(int)

    # lowRemove[val], highRemove[val] track how many times 'val' has been
    # flagged for removal from respective heaps (lazy removal approach).
    lowRemove = defaultdict(int)
    highRemove = defaultdict(int)

    # Keep track of the sum of all elements in "high" (the top K).
    top_sum = 0

    # A helper to remove invalid (stale) top elements of "low"
    # i.e. those that have a remove-flag or are no longer in freqLow
    def fix_low_top():
        # Pop while top is stale
        while low:
            val = -low[0]  # since stored as negative
            # If this val is flagged for removal
            if lowRemove[val] > 0:
                # discard one instance of val
                heapq.heappop(low)
                lowRemove[val] -= 1
            # or if freqLow[val] == 0, it's not in "low" anymore
            elif freqLow[val] == 0:
                heapq.heappop(low)
            else:
                # valid top
                break

    # Similarly for "high"
    def fix_high_top():
        while high:
            val = high[0]
            if highRemove[val] > 0:
                heapq.heappop(high)
                highRemove[val] -= 1
            elif freqHigh[val] == 0:
                heapq.heappop(high)
            else:
                break

    # Pop one valid item from the top of "low" (which is the maximum in low)
    # returns that value, or None if empty
    def pop_low():
        fix_low_top()
        if not low:
            return None
        val = -heapq.heappop(low)
        freqLow[val] -= 1
        if freqLow[val] == 0:
            del freqLow[val]
        return val

    # Pop one valid item (the smallest in high) from "high"
    def pop_high():
        fix_high_top()
        if not high:
            return None
        val = heapq.heappop(high)
        freqHigh[val] -= 1
        if freqHigh[val] == 0:
            del freqHigh[val]
        return val

    # Push a value into "low"
    def push_low(val):
        freqLow[val] += 1
        heapq.heappush(low, -val)

    # Push a value into "high"
    def push_high(val):
        freqHigh[val] += 1
        heapq.heappush(high, val)

    # Rebalance so that "high" has exactly K elements
    # Guarantee that every element in "high" is >= any element in "low".
    def rebalance():
        nonlocal top_sum
        # If high has too many elements, pop from high -> push to low
        while True:
            fix_high_top()
            if len(freqHigh) == 0:
                high_count = 0
            else:
                high_count = sum(freqHigh.values())
            if high_count <= K:
                break
            # pop from high
            val = pop_high()
            if val is None:  # no more valid in high
                break
            top_sum -= val
            # push into low
            push_low(val)

        # If high has too few elements, pop from low -> push to high
        while True:
            fix_low_top()
            if len(freqHigh) == 0:
                high_count = 0
            else:
                high_count = sum(freqHigh.values())
            if high_count >= K:
                break
            val = pop_low()
            if val is None:
                break
            top_sum += val
            push_high(val)

        # Make sure that the smallest element in "high" is >=
        # the largest element in "low" (if both sides are nonempty).
        # If not, we swap them.
        fix_low_top()
        fix_high_top()
        if low and high:
            # top of low is the largest in low
            max_in_low = -low[0]
            # top of high is the smallest in high
            min_in_high = high[0]
            if max_in_low > min_in_high:
                # swap
                # pop both
                big_val = pop_low()
                small_val = pop_high()
                # adjust top_sum
                top_sum -= small_val
                top_sum += big_val
                # push them cross
                push_low(small_val)
                push_high(big_val)

    # Initially, all A[i] = 0. We need to place them so that K zeros go to "high",
    # and the rest go to "low".
    # If K == N, all go to "high". Otherwise, we do the standard approach:
    for _ in range(N):
        push_low(0)
    top_sum = 0
    rebalance()  # so that we have exactly K zeros in "high".

    out = []
    for _ in range(Q):
        x = read_int() - 1  # 1-based to 0-based
        y = read_int()
        old_val = A[x]
        A[x] = y

        # Remove old_val from whichever place it was
        # We'll check freqHigh first. If freqHigh[old_val] > 0, it means
        # old_val is in high at least once.
        # Because each index's value can only account for 1 occurrence in the data structure,
        # we assume exactly one copy is in either high or low.
        # So we remove exactly 1 occurrence from whichever set has that value.
        if freqHigh[old_val] > 0:
            # remove from high
            freqHigh[old_val] -= 1
            if freqHigh[old_val] == 0:
                del freqHigh[old_val]
            highRemove[old_val] += 1
            top_sum -= old_val
        else:
            # remove from low
            freqLow[old_val] -= 1
            if freqLow[old_val] == 0:
                del freqLow[old_val]
            lowRemove[old_val] += 1

        # Insert new_val
        new_val = y
        # Decide if it goes to high or low
        fix_high_top()
        if len(freqHigh) < K:
            # We can put it directly into high
            push_high(new_val)
            top_sum += new_val
        else:
            # Compare with the smallest in high
            smallest_in_high = None
            if high:
                smallest_in_high = high[0]
            if smallest_in_high is not None and new_val > smallest_in_high:
                # Goes to high
                push_high(new_val)
                top_sum += new_val
                # Then we must pop one from high (the smallest) into low,
                # since we only want exactly K in high
                val_moved = pop_high()
                # val_moved is guaranteed to be new_val or the smallest, but we pop
                # immediately after pushing, so actually we might pop the smallest
                # whichever is the min. If that is new_val, we do not double insert.
                # We'll see:
                if val_moved is not None:
                    top_sum -= val_moved
                    push_low(val_moved)
            else:
                # Goes to low
                push_low(new_val)

        rebalance()

        # Now top_sum is f(A)
        out.append(str(top_sum))

    print("
".join(out))

# Do not forget to call main()!
if __name__ == "__main__":
    main()