def main():
    import sys
    # Use sys.stdin.buffer for fast input.
    data = sys.stdin.buffer.read().split()
    if not data:
        return

    n = int(data[0])
    # Read the sequence A as integers.
    A = [int(x) for x in data[1:1+n]]
    
    # Coordinate compression: sort unique values.
    unique_vals = sorted(set(A))
    M = len(unique_vals)
    comp = {val: idx + 1 for idx, val in enumerate(unique_vals)}  # 1-indexed for BIT

    # Initialize two Binary Indexed Trees (Fenwick Trees):
    # one for frequency counts and one for cumulative sums.
    bit_count = [0] * (M + 1)
    bit_sum = [0] * (M + 1)
    
    # BIT query: returns the prefix sum up to index i.
    def bit_query(bit, i):
        s = 0
        while i:
            s += bit[i]
            i -= i & -i
        return s

    # BIT update: adds delta to index i.
    def bit_update(bit, i, delta):
        while i <= M:
            bit[i] += delta
            i += i & -i

    ans = 0
    # Process each element in the sequence.
    for a in A:
        # Get the compressed rank for the current value.
        r = comp[a]
        # Query BITs for information about all earlier elements with value < a.
        # cnt is the count and sm is the sum of those elements.
        cnt = bit_query(bit_count, r - 1)
        sm = bit_query(bit_sum, r - 1)
        # For each such earlier element the contribution is (a - previous_value)
        ans += a * cnt - sm
        # Update BITs: record the current element.
        bit_update(bit_count, r, 1)
        bit_update(bit_sum, r, a)
    
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()