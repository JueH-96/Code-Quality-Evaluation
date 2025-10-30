def main():
    import sys
    input = sys.stdin.readline

    # Read N and Q
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))

    # Frequency array for values 0..N (anything >N does not affect mex up to N)
    freq = [0] * (N + 1)
    for x in A:
        if x <= N:
            freq[x] += 1

    # B[i] = 1 if freq[i] == 0 else 0, for i in [0..N]
    B = [0] * (N + 1)
    for i in range(N + 1):
        if freq[i] == 0:
            B[i] = 1

    # Fenwick (Binary Indexed) Tree to maintain prefix sums of B
    fenwicksum = [0] * (N + 2)

    def fenwicksum_update(pos, val):
        """Point update in Fenwick tree (1-based index)."""
        while pos <= N + 1:
            fenwicksum[pos] += val
            pos += pos & -pos

    # Build Fenwick tree in O(N)
    # First copy B into fenwicksum (shifting index by +1),
    # then run the Fenwick "build" pass.
    for i in range(N + 1):
        fenwicksum[i + 1] = B[i]
    for i in range(1, N + 2):
        j = i + (i & -i)
        if j <= N + 1:
            fenwicksum[j] += fenwicksum[i]

    def fenwicksum_find():
        """
        Returns the smallest index i in [0..N] such that
        prefix sum up to i (i.e. B[0] + ... + B[i]) >= 1.
        This corresponds to the mex if B[i] = 1 means freq[i] == 0.
        """
        pos = 0
        s = 1
        bit = 1 << (N + 1).bit_length()
        while bit > 0:
            nxt = pos + bit
            if nxt <= N + 1 and fenwicksum[nxt] < s:
                s -= fenwicksum[nxt]
                pos = nxt
            bit >>= 1
        return pos  # 0-based index

    def update_freq(v, delta):
        """
        Updates freq[v] by delta, and accordingly updates the Fenwick tree
        if freq[v] transitions between 0 and 1.
        """
        if v > N:
            # Values above N do not affect the mex calculation
            return
        old_val = freq[v]
        new_val = old_val + delta
        freq[v] = new_val
        # If freq changes from 0->1 or 1->0, we update B[v] and Fenwicksum.
        # B[v] = 1 if freq[v] == 0, else 0
        if old_val == 0 and new_val == 1:
            # B[v] changes from 1 to 0
            fenwicksum_update(v + 1, -1)
        elif old_val == 1 and new_val == 0:
            # B[v] changes from 0 to 1
            fenwicksum_update(v + 1, 1)

    # Process queries
    out = []
    for _ in range(Q):
        i, x = map(int, input().split())
        i -= 1  # make 0-based

        old_val = A[i]
        update_freq(old_val, -1)  # remove old_val
        A[i] = x
        update_freq(x, 1)        # add new val

        # Compute mex = smallest i with freq[i] == 0
        mex_index = fenwicksum_find()
        out.append(str(mex_index))

    print("
".join(out))


# Do not forget to call main() at the end
if __name__ == "__main__":
    main()