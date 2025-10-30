def main():
    import sys
    import math
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    q = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # We only care about numbers in [0, n+1]
    M = n + 2  # range size. We'll maintain counts for j in [0, n+1]
    counts = [0] * (M)
    for val in A:
        if 0 <= val < M:
            counts[val] += 1
    
    # Build an iterative segment tree for the range [0, M)
    size = 1
    while size < M:
        size *= 2
    # seg will have size 2*size, leaves are at index [size, size+M-1]
    seg = [0] * (2 * size)
    
    # function to recalc a node
    def recalc(i):
        seg[i] = seg[2*i] + seg[2*i+1]
    
    # Build leaves:
    for i in range(M):
        # if counts[i]==0 then leaf value is 1, else 0.
        seg[size + i] = 1 if counts[i] == 0 else 0
    for i in range(M, size):
        seg[size + i] = 0
    # Build internal nodes:
    for i in range(size - 1, 0, -1):
        recalc(i)
    
    # update function: update the value at position pos to value 'val'
    def update(pos, val):
        i = size + pos
        seg[i] = val
        i //= 2
        while i:
            recalc(i)
            i //= 2
    
    # query: find smallest index j in [0, M) with seg value 1.
    def query():
        i = 1
        if seg[i] == 0:
            return M  # though this should not happen because mex always in [0, M)
        while i < size:
            if seg[2*i] > 0:
                i = 2*i
            else:
                i = 2*i + 1
        return i - size

    # helper to update frequency counts and update segment tree leaves accordingly
    def change_count(num, delta):
        # Only care about num in [0, M)
        if 0 <= num < M:
            old = counts[num]
            counts[num] += delta
            new = counts[num]
            # if old==0 and now becomes >0 then remove missing marker
            if old == 0 and new > 0:
                update(num, 0)
            # if old>0 and now becomes 0 then add missing marker
            if old > 0 and new == 0:
                update(num, 1)
    
    out_lines = []
    # Process each query:
    for _ in range(q):
        i_str = next(it)
        x_str = next(it)
        idx = int(i_str) - 1  # zero-indexed
        new_val = int(x_str)
        old_val = A[idx]
        # Update frequency for old and new values.
        change_count(old_val, -1)
        change_count(new_val, +1)
        A[idx] = new_val
        # Answer the query
        mex = query()
        out_lines.append(str(mex))
    
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()