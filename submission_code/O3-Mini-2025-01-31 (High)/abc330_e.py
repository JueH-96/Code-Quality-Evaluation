def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    M = N + 1  # We only care about candidate values 0..N (mex is always in [0,N] for an array of length N)

    # Build frequency for numbers in the range [0, N]
    freq = [0] * M
    for a in A:
        if 0 <= a < M:
            freq[a] += 1

    # Build an indicator array: arr[i] == 1 if i is missing from A, else 0.
    arr = [1 if freq[i] == 0 else 0 for i in range(M)]

    # Build Fenwick Tree (Binary Indexed Tree) over the indicator array.
    ft = [0] * (M + 1)
    for i in range(M):
        ft[i + 1] = arr[i]
    for i in range(1, M + 1):
        j = i + (i & -i)
        if j <= M:
            ft[j] += ft[i]

    # Update function for the Fenwick Tree.
    def fenw_update(i, delta):
        idx = i + 1  # use 1-indexed positions in the tree
        while idx <= M:
            ft[idx] += delta
            idx += idx & -idx

    # Returns the smallest index i (0-indexed) such that the prefix sum is at least x.
    def fenw_lower_bound(x):
        pos = 0
        bit = 1 << (M.bit_length() - 1)
        while bit:
            nxt = pos + bit
            if nxt <= M and ft[nxt] < x:
                x -= ft[nxt]
                pos = nxt
            bit //= 2
        return pos

    # Process each query.
    out_lines = []
    for _ in range(Q):
        idx = int(next(it)) - 1  # Convert query index from 1-indexed to 0-indexed.
        new_val = int(next(it))
        old_val = A[idx]
        if old_val == new_val:
            # No change needed; simply report the current mex.
            mex = fenw_lower_bound(1)
            out_lines.append(str(mex))
            continue

        # Remove the effect of the old value (if it falls in the [0, N] range).
        if 0 <= old_val < M:
            freq[old_val] -= 1
            if freq[old_val] == 0:
                # It just became missing, so mark it in our Fenwick tree.
                fenw_update(old_val, 1)
        
        # Add the effect of the new value (if it falls in the [0, N] range).
        if 0 <= new_val < M:
            if freq[new_val] == 0:
                # It was previously missing; now it will be present.
                fenw_update(new_val, -1)
            freq[new_val] += 1
        
        A[idx] = new_val
        # The answer (mex) is the smallest number i such that index i is missing.
        mex = fenw_lower_bound(1)
        out_lines.append(str(mex))
        
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()