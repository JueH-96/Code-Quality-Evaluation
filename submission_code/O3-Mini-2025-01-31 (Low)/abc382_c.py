def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(m)]
    
    # We want to answer queries of the form:
    # "Find the smallest index i (0-indexed) such that A[i] <= sushi_deliciousness"
    # Since each query is independent (each sushi travels from person 1 to person N),
    # we can preprocess a segment tree to answer these queries.
    #
    # We'll build a segment tree over A where each node holds the minimum value in that segment.
    # Then for a given sushi with deliciousness x, if the global minimum is > x no person qualifies.
    # Otherwise, we perform a binary search on the tree (starting from the root)
    # to locate the leftmost index with A[i] <= x.
    
    # Build segment tree: use a tree array of size 2*size, where size is a power of 2 >= n.
    size = 1
    while size < n:
        size *= 2
    seg = [10**9] * (2 * size)
    
    # Fill leaves
    for i in range(n):
        seg[size + i] = A[i]
    # Build internal nodes
    for i in range(size - 1, 0, -1):
        seg[i] = seg[2*i] if seg[2*i] < seg[2*i+1] else seg[2*i+1]
    
    # Function to get the leftmost index i (0-indexed) with A[i] <= x
    def query(x):
        # If the minimum in the whole array is > x, no person qualifies.
        if seg[1] > x:
            return -1
        idx = 1
        # Descend the tree until we reach a leaf.
        while idx < size:
            # Check the left child. If it qualifies, go left.
            if seg[2 * idx] <= x:
                idx = 2 * idx
            else:
                idx = 2 * idx + 1
        # Convert tree leaf index to array index.
        return idx - size
    
    # Process each sushi in order.
    out_lines = []
    for val in B:
        i = query(val)
        if i == -1:
            out_lines.append("-1")
        else:
            # Convert 0-indexed to 1-indexed person number.
            out_lines.append(str(i + 1))
    
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()