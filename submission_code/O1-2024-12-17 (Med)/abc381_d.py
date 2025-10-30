def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Edge case: if N < 2, the longest 1122 subarray (possibly empty) is 0
    if N < 2:
        print(0)
        return

    # 1) Identify all valid pairs (i, i+1) where A[i] == A[i+1].
    #    We'll store (i, value) for such pairs, using 0-based index i.
    pairs = []
    for i in range(N - 1):
        if A[i] == A[i + 1]:
            pairs.append((i, A[i]))

    # If there are no pairs, answer is 0 (empty subarray).
    if not pairs:
        print(0)
        return

    # 2) We want the longest contiguous chain of such pairs
    #    where consecutive pairs are exactly 2 indices apart (pos[i+1] = pos[i] + 2)
    #    and the values are all distinct within that chain.
    # We will use a two-pointer/sliding window approach over the array "pairs".
    best = 1  # At least one pair (if any) forms a valid subarray of length 2
    start = 0
    used = {pairs[0][1]}  # Keep track of values in the current chain

    for i in range(1, len(pairs)):
        pos_current, val_current = pairs[i]
        pos_prev, val_prev = pairs[i - 1]

        # Check if we can extend the chain
        #  - Positions must be adjacent in steps of 2
        #  - The new value must not be in 'used'
        if pos_current == pos_prev + 2 and val_current not in used:
            # Extend the chain
            used.add(val_current)
        else:
            # Current chain breaks here. Update best, then start a new chain at i.
            best = max(best, i - start)
            start = i
            used = {val_current}

    # Check the last chain
    best = max(best, len(pairs) - start)

    # Each pair contributes length 2 in the array
    print(best * 2)

# Do not forget to call main()
if __name__ == "__main__":
    main()