def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Coordinate compression
    unique_values = sorted(set(A))
    # A dictionary that maps the original value to the compressed one
    # We'll map them into 1-based indices for Fenwick convenience
    comp = {}
    for i, v in enumerate(unique_values, start=1):
        comp[v] = i

    # Fenwick Tree (BIT) for counts
    # Fenwick Tree (BIT) for sums
    size = len(unique_values)
    fenw_count = [0]*(size+1)
    fenw_sum = [0]*(size+1)

    # Fenwicks typical definitions
    def fenw_update(bit, idx, val):
        while idx <= size:
            bit[idx] += val
            idx += idx & -idx
    
    def fenw_query(bit, idx):
        s = 0
        while idx > 0:
            s += bit[idx]
            idx -= idx & -idx
        return s

    ans = 0
    for x in A:
        # Compressed index
        cx = comp[x]
        # Query how many are strictly less than x -> query(cx-1)
        c = fenw_query(fenw_count, cx-1)   # count of values less than x
        s = fenw_query(fenw_sum, cx-1)    # sum of values less than x
        ans += x*c - s
        # Update Fenwicks
        fenw_update(fenw_count, cx, 1)
        fenw_update(fenw_sum, cx, x)
    
    print(ans)

# Do not forget to call main()!
if __name__ == "__main__":
    main()