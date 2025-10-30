import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = list(map(int, (next(it) for _ in range(N))))
    B = list(map(int, (next(it) for _ in range(M))))
    
    # Segment tree for range minimum, 0-based on A[0..N-1]
    size = 1
    while size < N:
        size <<= 1
    INF = 10**18
    seg = [INF] * (2 * size)
    
    # Build leaves
    for i in range(N):
        seg[size + i] = A[i]
    # Build internal nodes
    for i in range(size - 1, 0, -1):
        seg[i] = min(seg[2*i], seg[2*i+1])
    
    # find_first: find smallest index i in [l..r] where A[i] <= x, or -1
    def find_first(x, idx, left, right):
        # Node idx covers [left, right)
        if seg[idx] > x:
            return -1
        if right - left == 1:
            # leaf
            return left
        mid = (left + right) // 2
        # check left child
        res = find_first(x, 2*idx, left, mid)
        if res != -1:
            return res
        return find_first(x, 2*idx+1, mid, right)
    
    out = []
    for b in B:
        pos = find_first(b, 1, 0, size)
        if pos == -1 or pos >= N:
            out.append("-1")
        else:
            out.append(str(pos + 1))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()