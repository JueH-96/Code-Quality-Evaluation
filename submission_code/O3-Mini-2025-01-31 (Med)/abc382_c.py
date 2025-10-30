def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    sushi_values = [int(next(it)) for _ in range(m)]
    
    # Build an iterative segment tree for range minimum query.
    # We'll find the leftmost index i such that A[i] <= s for each sushi value s.
    size = 1
    while size < n:
        size *= 2
    seg = [10**9] * (2 * size)  # use a large sentinel value for initialization
    
    # Set leaves (indices from size to size+n-1)
    for i in range(n):
        seg[size + i] = A[i]
        
    # Build the segment tree upwards.
    for i in range(size - 1, 0, -1):
        # seg[i] is the minimum value in its children
        left = seg[2 * i]
        right = seg[2 * i + 1]
        seg[i] = left if left < right else right
    
    # For each sushi piece, determine who eats it.
    # A sushi of deliciousness s is eaten by the first person i (in order)
    # with A[i] <= s.  If no such person exists, output -1.
    out_lines = []
    for s in sushi_values:
        if seg[1] > s:
            out_lines.append("-1")
            continue
        idx = 1
        # Traverse the segment tree to find the leftmost leaf with A[i] <= s.
        while idx < size:
            # If the left child's min is <= s, then we know an eligible person exists there.
            if seg[2 * idx] <= s:
                idx = 2 * idx
            else:
                idx = 2 * idx + 1
        # idx is now the leaf index; convert to person number (1-indexed)
        person = idx - size + 1
        out_lines.append(str(person))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()