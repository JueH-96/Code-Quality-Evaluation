import sys
import threading
def main():
    import sys
    
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(m)]
    
    # Build a segment tree that stores the minimum A_i in each segment.
    size = 1
    while size < n:
        size <<= 1
    INF = 10**18
    seg = [INF] * (2 * size)
    
    # Initialize leaves
    for i in range(n):
        seg[size + i] = A[i]
    # Build internal nodes
    for i in range(size - 1, 0, -1):
        seg[i] = min(seg[2*i], seg[2*i + 1])
    
    # Function to find the first index i (0-based) such that A[i] <= x, or -1 if none.
    # We know seg[1] <= x when we call this, else we return -1 immediately.
    def find_first(x, node, nl, nr):
        if nl == nr:
            # Leaf
            return nl
        mid = (nl + nr) // 2
        left = node * 2
        # If left child's minimum is <= x, go left
        if seg[left] <= x:
            return find_first(x, left, nl, mid)
        else:
            # Otherwise go right
            return find_first(x, left + 1, mid + 1, nr)
    
    out = []
    for x in B:
        # If overall minimum > x, nobody can eat this sushi
        if seg[1] > x:
            out.append("-1")
        else:
            idx = find_first(x, 1, 0, size - 1)
            # idx might be >= n when size > n; but seg values there are INF, so won't be chosen
            out.append(str(idx + 1))  # convert to 1-based
    
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()