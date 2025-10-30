def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # To speed up output in Python:
    import sys
    write_out = sys.stdout.write

    # Parsing inputs
    N, Q = map(int, input_data[:2])
    S = input_data[2]
    queries = input_data[3:]

    # Fenwick (Binary Indexed) Tree for point‐update & range‐query of array B,
    # where B[i] = 1 if S[i] == S[i+1], else 0 (0-based i from 0..N-2).
    # For a substring check [L..R] in 1-based indices, we actually need
    # sum of B in [L-1 .. R-2] (0-based) = number of "equal‐adjacent" pairs
    # in that substring. If the sum is 0 => "Yes", else "No".
    #
    # Flipping [L..R] in 1-based indexing toggles exactly two edges in B at most:
    #   - Edge L-2 (if L>=2)   (0-based)  => compares S[L-2], S[L-1]
    #   - Edge R-1 (if R<N)    (0-based)  => compares S[R-1], S[R]
    # Because a flip only changes adjacency if exactly one of the two indices
    # in a pair is flipped.
    #
    # We will keep track of B in a Fenwick tree:
    #   - fenwicksum(x) gives sum B[0..x]
    #   - fenwickadd(pos, val) adds val to B[pos]
    #
    # Then for a query type 2 (check substring [L..R]):
    # we compute sumB = range_sum_B(L-1, R-2).
    # If sumB == 0 => "Yes", else "No".

    # Build array B of length N-1 (0-based).
    # B[i] = 1 if S[i] == S[i+1], else 0.
    B = [0]*(N-1)
    for i in range(N-1):
        if S[i] == S[i+1]:
            B[i] = 1

    # Fenwick tree implementation
    # We'll build it from array B.
    fenwicksz = N-1
    fenw = [0]*(fenwicksz+1)  # 1-based Fenwick internally

    def fenwick_add(idx, val):
        # idx is 0-based for B, fenw is 1-based internally
        i = idx+1
        while i <= fenwicksz:
            fenw[i] += val
            i += i & -i

    def fenwick_sum(idx):
        # sum from B[0]..B[idx]
        # idx is 0-based
        s = 0
        i = idx+1
        while i>0:
            s += fenw[i]
            i -= i & -i
        return s

    def fenwick_range_sum(left, right):
        # sum B[left..right], inclusive
        if left>right:
            return 0
        return fenwick_sum(right) - fenwick_sum(left-1) if left>0 else fenwick_sum(right)

    # Build Fenwick from B
    for i, val in enumerate(B):
        fenwick_add(i, val)

    # Function to toggle B[i] (from 0->1 or 1->0) and update Fenwicks
    def toggle_b(i):
        # i must be between 0 and N-2 inclusive
        if 0 <= i < (N-1):
            old_val = B[i]
            new_val = 1 - old_val
            B[i] = new_val
            fenwick_add(i, new_val - old_val)

    # Process queries
    # We read them in blocks of three from queries array:
    # index usage: each query has [type, L, R]
    idx = 0
    out = []
    for _ in range(Q):
        t = int(queries[idx]); idx+=1
        L = int(queries[idx]); idx+=1
        R = int(queries[idx]); idx+=1
        # Convert to 0-based where needed
        if t == 1:
            # Flip [L..R], 1-based => check boundary edges in B
            # i.e. toggle B[L-2] if L>=2
            #      toggle B[R-1] if R<N
            if L >= 2:
                toggle_b(L-2)
            if R < N:
                toggle_b(R-1)
        else:
            # t == 2 => check substring [L..R]
            # sum B in [L-1..R-2] (in 0-based).
            left_idx = L-1
            right_idx = R-2
            cnt = fenwick_range_sum(left_idx, right_idx)
            if cnt == 0:
                out.append("Yes")
            else:
                out.append("No")

    write_out("
".join(out) + "
")

# Call solve()
def main():
    solve()

if __name__ == "__main__":
    main()