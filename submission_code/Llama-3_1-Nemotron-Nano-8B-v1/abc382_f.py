import sys
from math import log2, ceil

def main():
    sys.setrecursionlimit(1 << 25)
    H, W, N = map(int, sys.stdin.readline().split())
    bars = []
    for _ in range(N):
        R, C, L = map(int, sys.stdin.readline().split())
        bars.append((R-1, C-1, L))  # 0-based indices

    # We process bars in reverse order
    bars.reverse()

    # Segment Tree for range max query and range update
    class SegmentTree:
        def __init__(self, size):
            self.n = 1 << (ceil(log2(size)) if size else 0)
            self.size = size
            self.tree = [0] * (2 * self.n)
        
        def update_range(self, l, r, val):
            l += self.n
            r += self.n
            while l <= r:
                if l % 2 == 1:
                    self.tree[l] = max(self.tree[l], val)
                    l += 1
                if r % 2 == 0:
                    self.tree[r] = max(self.tree[r], val)
                    r -= 1
                l //= 2
                r //= 2
        
        def query_range(self, l, r):
            res = 0
            l += self.n
            r += self.n
            while l <= r:
                if l % 2 == 1:
                    res = max(res, self.tree[l])
                    l += 1
                if r % 2 == 0:
                    res = max(res, self.tree[r])
                    r -= 1
                l //= 2
                r //= 2
            return res

    st = SegmentTree(W)

    res = []
    for R, C, L in bars:
        C_end = C + L - 1
        if C_end >= W:
            res.append((R+1,))  # 1-based output, but this case should not happen per input constraints
            continue
        max_row = st.query_range(C, C_end)
        if max_row == 0:
            R_i = R + (H - R - 1)
        else:
            R_i = min(R + (max_row - R), H - L + 1)
        res.append(R_i + 1)  # convert back to 1-based
        st.update_range(C, C_end, R_i + 1)  # store the row where the bar ends, which is R_i + 1 (1-based)

    for ans in reversed(res):
        print(ans)

if __name__ == '__main__':
    main()