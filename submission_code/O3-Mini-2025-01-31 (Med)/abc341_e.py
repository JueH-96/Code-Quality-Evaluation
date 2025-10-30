def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Use an iterator that yields tokens (all are bytes).
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    s_line = next(it).decode().strip()
    # s_orig stores the original bits (0 or 1) in a 0-indexed list.
    s_orig = [1 if ch == '1' else 0 for ch in s_line]

    # We want to support range "flip" queries on the string efficiently.
    # We do this by maintaining a Fenwick (BIT) structure (flip_bit)
    # for the cumulative number of flips at each position (mod 2).
    # Then the effective value at position i (1-indexed) is:
    #    effective_value(i) = s_orig[i-1] XOR (flip_prefix(i) mod 2)
    #
    # Also, a string is "good" if every pair of consecutive characters are different.
    # We maintain an auxiliary array (of size n-1) for the pairs.
    # For each index i from 1 to n-1, let violation[i] = 1 if
    # the effective values at positions i and i+1 are the same (i.e. a violation),
    # and 0 otherwise.
    # Then a substring [L, R] is good if and only if the sum of violation[i] for i from L to R-1 is 0.
    #
    # Notice that when we flip a range [L,R], the effective bits for indices L..R are toggled.
    # However, for any adjacent pair completely inside or completely outside the range,
    # both endpoints get toggled or none gets toggled so their equality does not change.
    # Only pairs crossing the boundary (at L-1,L and at R,R+1) might change.
    #
    # We use a BIT for range updates ("flip_bit") and another BIT for point updates on the
    # violation array ("vio_bit"). With this design, each query (flip or check) will be O(log n).
    #
    # BIT for range-flip updates (1-indexed, size = n+1):
    size = n + 1
    flip_bit = [0] * size

    def flip_update(i, delta):
        # update BIT at position i (1-indexed)
        while i < size:
            flip_bit[i] += delta
            i += i & -i

    def flip_query(i):
        # returns the prefix sum mod 2 at position i (1-indexed)
        s = 0
        while i:
            s += flip_bit[i]
            i -= i & -i
        return s & 1

    # BIT for violations. We have violation indices from 1 to n-1.
    vio_size = n  # We use indices 1..n-1 (index 0 is unused).
    vio_bit = [0] * vio_size
    # We also keep an array for the current violation value at each index i (1-indexed, for i in 1..n-1).
    vio_arr = [0] * n

    def vio_update(i, delta):
        # point update for the violation BIT at position i (1-indexed)
        j = i
        while j < vio_size:
            vio_bit[j] += delta
            j += j & -j

    def vio_query(i):
        s = 0
        while i:
            s += vio_bit[i]
            i -= i & -i
        return s

    def vio_range_query(l, r):
        return vio_query(r) - vio_query(l - 1)

    # Given a 1-indexed position i, compute the effective value.
    def effective_val(i):
        # Effective value = s_orig[i-1] XOR (flip prefix mod 2)
        return s_orig[i - 1] ^ flip_query(i)

    # Build the violation BIT from the initial string.
    # For every adjacent pair (i, i+1) for i = 1 to n-1, if s_orig[i-1] equals s_orig[i] then that pair is a violation.
    for i in range(1, n):
        if s_orig[i - 1] == s_orig[i]:
            vio_arr[i] = 1
        else:
            vio_arr[i] = 0
        j = i
        while j < vio_size:
            vio_bit[j] += vio_arr[i]
            j += j & -j

    out_lines = []
    # Process the queries in order.
    for _ in range(q):
        typ = next(it).decode()
        if typ == '1':
            # Query type 1: flip range L R.
            L = int(next(it))
            R = int(next(it))
            # Range update: add 1 flip to indices L..R.
            flip_update(L, 1)
            if R + 1 <= n:
                flip_update(R + 1, -1)
            # Only two boundaries may have changed:
            # (L-1, L) if L > 1, and (R, R+1) if R < n.
            if L > 1:
                # i = L-1 corresponds to pair (L-1, L)
                idx = L - 1
                old_val = vio_arr[idx]
                new_val = 1 if effective_val(L - 1) == effective_val(L) else 0
                if new_val != old_val:
                    diff = new_val - old_val
                    vio_arr[idx] = new_val
                    vio_update(idx, diff)
            if R < n:
                # i = R corresponds to pair (R, R+1)
                idx = R
                old_val = vio_arr[idx]
                new_val = 1 if effective_val(R) == effective_val(R + 1) else 0
                if new_val != old_val:
                    diff = new_val - old_val
                    vio_arr[idx] = new_val
                    vio_update(idx, diff)
        else:
            # Query type 2: Check whether the substring S[L..R] is good.
            L = int(next(it))
            R = int(next(it))
            if L == R:
                out_lines.append("Yes")
            else:
                # The substring is good if every adjacent pair from L to R-1 is different,
                # i.e. the sum of violation values in [L, R-1] is 0.
                if vio_range_query(L, R - 1) == 0:
                    out_lines.append("Yes")
                else:
                    out_lines.append("No")
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()