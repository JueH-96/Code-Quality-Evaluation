def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast I/O setup:
    # input_data = []
    # for line in sys.stdin:
    #     input_data.extend(line.split())

    # Parse initial input
    N, Q = map(int, input_data[:2])
    S_str = input_data[2]
    # 1-based indexing for original bits
    originalS = [0]*(N+1)
    for i in range(N):
        originalS[i+1] = 1 if S_str[i] == '1' else 0

    # Fenwick (Binary Indexed) Tree for range-flip (XOR) operations:
    # flipFenwicks[i] will store partial XOR increments.
    # To flip [L..R], we do update(L,1) and update(R+1,1) (if R+1 <= N).
    # The actual bit at position i is originalS[i] ^ fenwicksXOR_get(i).
    fenwicksFlip = [0]*(N+1)

    def fenwicksXOR_update(pos, val):
        while pos <= N:
            fenwicksFlip[pos] ^= val
            pos += (pos & -pos)

    def fenwicksXOR_get(pos):
        s = 0
        while pos > 0:
            s ^= fenwicksFlip[pos]
            pos -= (pos & -pos)
        return s

    def getBit(i):
        # Returns the current bit at position i (1-based)
        # after taking all flips into account.
        return originalS[i] ^ fenwicksXOR_get(i)

    # We create D[i] = 1 if S[i] != S[i+1], else 0, for i=1..N-1.
    # Then the substring S[L..R] is "good" iff sum(D[L..R-1]) == (R-L).
    D = [0]*(N+1)
    for i in range(1, N):
        D[i] = 1 if originalS[i] != originalS[i+1] else 0

    # Fenwicks for sums of D (to count how many "diff" in a range).
    fenwicksD = [0]*(N+1)

    def fenwicksSum_update(pos, delta):
        while pos <= N:
            fenwicksD[pos] += delta
            pos += (pos & -pos)

    def fenwicksSum_get(pos):
        s = 0
        while pos > 0:
            s += fenwicksD[pos]
            pos -= (pos & -pos)
        return s

    # Build the fenwicks tree for D
    for i in range(1, N):
        if D[i] == 1:
            fenwicksSum_update(i, 1)

    # Process queries
    idx = 3
    out = []
    for _ in range(Q):
        t = int(input_data[idx]); idx += 1
        L = int(input_data[idx]); idx += 1
        R = int(input_data[idx]); idx += 1

        if t == 1:
            # Flip S[L..R]
            fenwicksXOR_update(L, 1)
            if R+1 <= N:
                fenwicksXOR_update(R+1, 1)

            # Only adjacencies at L-1 and R can change
            # 1) D[L-1] if L > 1
            if L > 1:
                old_val = D[L-1]
                left_bit = getBit(L-1)
                curr_bit = getBit(L)
                new_val = 1 if left_bit != curr_bit else 0
                D[L-1] = new_val
                if new_val != old_val:
                    fenwicksSum_update(L-1, new_val - old_val)

            # 2) D[R] if R < N
            if R < N:
                old_val = D[R]
                left_bit = getBit(R)
                curr_bit = getBit(R+1)
                new_val = 1 if left_bit != curr_bit else 0
                D[R] = new_val
                if new_val != old_val:
                    fenwicksSum_update(R, new_val - old_val)

        else:
            # t == 2; check if S[L..R] is a good string
            if L == R:
                # Single character is always good
                out.append("Yes")
            else:
                # sum of D[L..R-1] should be (R-L) if all adjacents differ
                sdiff = fenwicksSum_get(R-1) - fenwicksSum_get(L-1)
                out.append("Yes" if sdiff == (R - L) else "No")

    print("
".join(out))

# Do not forget to call main().
if __name__ == "__main__":
    main()