import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    data = sys.stdin
    N, Q = map(int, data.readline().split())
    A = list(map(int, data.readline().split()))
    # We'll track counts for values in [0..N+1]
    M = N + 2
    cnt = [0] * (M)
    # Count initial
    for v in A:
        if 0 <= v < M:
            cnt[v] += 1
    # Build segment tree over an array present[i] = 1 if cnt[i]>0 else 0
    size = 1
    while size < M:
        size <<= 1
    seg = [0] * (2 * size)
    # initialize leaves
    for i in range(M):
        seg[size + i] = 1 if cnt[i] > 0 else 0
    for i in range(size - 1, 0, -1):
        seg[i] = seg[2*i] + seg[2*i+1]
    # helper to update present at position p to val (0 or 1)
    def seg_update(p, val):
        i = p + size
        if seg[i] == val:
            return
        seg[i] = val
        i >>= 1
        while i:
            seg[i] = seg[2*i] + seg[2*i+1]
            i >>= 1
    # find first index in [0..M-1] with present==0
    def find_mex():
        # if entire [0..M) is full, mex would be M, but that's impossible since M=N+2 and mex<=N+1
        i = 1
        l = 0
        r = size
        # if sum==r-l then all present; but won't happen
        while i < size:
            left_sum = seg[2*i]
            mid = (l + r) >> 1
            # left segment length = mid-l
            if left_sum < (mid - l):
                # there's a zero in left child
                i = 2*i
                r = mid
            else:
                # go right
                i = 2*i+1
                l = mid
        # now leaf at i represents index l
        return l
    out = []
    for _ in range(Q):
        line = data.readline().split()
        if not line:
            break
        idx = int(line[0]) - 1
        x = int(line[1])
        old = A[idx]
        # decrement old
        if 0 <= old < M:
            cnt[old] -= 1
            if cnt[old] == 0:
                # becomes missing
                seg_update(old, 0)
        # increment new x
        if 0 <= x < M:
            if cnt[x] == 0:
                # was missing, now present
                seg_update(x, 1)
            cnt[x] += 1
        A[idx] = x
        # answer mex
        mex = find_mex()
        out.append(str(mex))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()