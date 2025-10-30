# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Read arrays A and B (0-indexed)
    A = [0] * n
    B = [0] * n
    for i in range(n):
        A[i] = int(next(it))
    for i in range(n):
        B[i] = int(next(it))
        
    # We represent each single index i by a pair (add, mul) where:
    #   - if B[i] == 1, the operation is f(x)= x + A[i]  →  (A[i], 1)
    #   - if B[i] > 1, the operation is f(x)= max(x + A[i], x * B[i])
    #     (its “switch‐threshold” is A[i]/(B[i]-1)) → we store it as (A[i], B[i]).
    #
    # The identity (do nothing) function is (0, 1) because f(x)= x.
    
    # Build an iterative segment tree.
    size = 1
    while size < n:
        size *= 2
    seg = [(0,1)] * (2*size)
    for i in range(n):
        b_val = B[i]
        if b_val <= 1:
            b_val = 1
        seg[size+i] = (A[i], b_val)
    for i in range(n, size):
        seg[size+i] = (0, 1)
    
    # The combine (merge) function. For f = (a, m) and g = (b, n)
    # (where f is applied first, then g) we want:
    #    result = g o f (so that result(0) = g(f(0)) )
    # Note: If f(0)= a then applying g gives:
    #    g(a) =  a + b       if a < (b/(n-1))   [when n > 1]
    #         = a * n       otherwise.
    # To avoid floats, we use the equivalent:
    #    if n==1 or a*(n-1) < b then new_add = a + b  else new_add = a * n.
    # And new_mul = m * n.
    def combine(f, g):
        a, m = f
        b, n = g
        if n == 1 or a*(n-1) < b:
            new_a = a + b
        else:
            new_a = a * n
        new_m = m * n
        return (new_a, new_m)
    
    # Build the tree.
    for i in range(size-1, 0, -1):
        seg[i] = combine(seg[2*i], seg[2*i+1])
    
    # Identity function for queries.
    ID = (0, 1)
    
    # Query function returns the composite function (as a tuple) for [l, r)
    def query(l, r):
        res_left = ID
        res_right = ID
        l += size
        r += size
        while l < r:
            if l & 1:
                res_left = combine(res_left, seg[l])
                l += 1
            if r & 1:
                r -= 1
                res_right = combine(seg[r], res_right)
            l //= 2
            r //= 2
        return combine(res_left, res_right)
    
    out_lines = []
    q = int(next(it))
    # Update function – update position pos (0-indexed) with new tuple value.
    def update(pos, v):
        i = pos + size
        seg[i] = v
        i //= 2
        while i:
            seg[i] = combine(seg[2*i], seg[2*i+1])
            i //= 2
    
    # Process queries.
    for _ in range(q):
        typ = next(it)
        if typ == b'1':  # update A_i
            i = int(next(it)) - 1
            x = int(next(it))
            A[i] = x
            b_val = B[i]
            if b_val <= 1:
                b_val = 1
            update(i, (x, b_val))
        elif typ == b'2':  # update B_i
            i = int(next(it)) - 1
            x = int(next(it))
            B[i] = x
            b_val = x
            if b_val <= 1:
                b_val = 1
            update(i, (A[i], b_val))
        else:  # query 3 l r
            l = int(next(it)) - 1
            r = int(next(it))
            # The composite function over [l, r] applied to 0 gives dp_results = add.
            res = query(l, r)
            out_lines.append(str(res[0]))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()