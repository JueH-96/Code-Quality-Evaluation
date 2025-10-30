def main():
    import sys,sys
    input_data = sys.stdin.buffer.read().split()
    if not input_data:
        return
    itr = iter(input_data)
    H = int(next(itr))
    W = int(next(itr))
    N = int(next(itr))
    bars = []
    for i in range(N):
        R = int(next(itr))
        C = int(next(itr))
        L = int(next(itr))
        bars.append((R, C, L, i))
    
    # We process bars in order of descending initial row.
    bars.sort(key = lambda x: x[0], reverse=True)
    
    # Segment Tree implementation with lazy propagation
    size = 1
    while size < W:
        size *= 2
    # tree stores the minimum value in the range.
    tree = [H] * (2*size)
    # lazy stores pending assignment (if not None, means assign that value to the node's range)
    lazy = [None] * (2*size)
    
    # Build procedure: already tree leaves set to H.
    # Our segment tree covers indices 0 .. size-1 mapping to columns 1..W (we use 0-indexed tree for convenience)
    # The first W leaves correspond to columns.
    def apply(node, val, node_left, node_right):
        tree[node] = val
        lazy[node] = val

    # Push the lazy value down to children
    def push(node, node_left, node_right):
        if lazy[node] is not None and node < size:  # has children
            mid = (node_left + node_right) // 2
            left_child = 2*node
            right_child = 2*node+1
            apply(left_child, lazy[node], node_left, mid)
            apply(right_child, lazy[node], mid+1, node_right)
            lazy[node] = None

    # Update assignment in range [l, r] (1-indexed positions will be mapped to [l-1, r-1])
    def update(node, node_left, node_right, l, r, val):
        if r < node_left or node_right < l:
            return
        if l <= node_left and node_right <= r:
            apply(node, val, node_left, node_right)
            return
        push(node, node_left, node_right)
        mid = (node_left + node_right)//2
        update(2*node, node_left, mid, l, r, val)
        update(2*node+1, mid+1, node_right, l, r, val)
        tree[node] = min(tree[2*node], tree[2*node+1])
    
    # Query minimum in range [l, r]
    def query(node, node_left, node_right, l, r):
        if r < node_left or node_right < l:
            return float('inf')
        if l <= node_left and node_right <= r:
            return tree[node]
        push(node, node_left, node_right)
        mid = (node_left + node_right)//2
        return min(query(2*node, node_left, mid, l, r),
                   query(2*node+1, mid+1, node_right, l, r))
    
    res = [None]*N
    # Process each bar
    for R, C, L, idx in bars:
        # our segment tree indices are 0-indexed: column C corresponds to index C-1
        left = C - 1
        right = (C + L - 1) - 1  # equals C+L-2
        candidate = query(1, 0, size-1, left, left+L-1)
        # The bar’s final row is the maximum of its initial row and candidate.
        final_r = R if R >= candidate else candidate
        res[idx] = final_r
        new_val = final_r - 1  # update support in these columns
        update(1, 0, size-1, left, left+L-1, new_val)
    
    sys.stdout.write("
".join(map(str, res)))
    
if __name__ == '__main__':
    main()