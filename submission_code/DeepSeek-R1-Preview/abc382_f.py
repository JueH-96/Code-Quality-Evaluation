import sys
from math import inf
import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx]); idx +=1
    W = int(data[idx]); idx +=1
    N = int(data[idx]); idx +=1

    bars = []
    for _ in range(N):
        R = int(data[idx])-1; idx +=1  # 0-based
        C = int(data[idx])-1; idx +=1
        L = int(data[idx]); idx +=1
        bars.append( (R, C, L) )

    # We'll use a segment tree to track the earliest blocked row for each column
    # Initially, all are H (since bars can fall to row H-1, which is the bottom)
    # Wait, no: The grid has rows 0 to H-1, and a bar can move to H-1.
    # So, the bottom is row H-1, and the 'floor' for each column is initially H.

    # Segment tree for range min query and point updates
    class SegmentTree:
        def __init__(self, size, default):
            self.n = 1
            while self.n < size:
                self.n <<=1
            self.size = size
            self.default = default
            self.tree = [default] * (2 * self.n)
        
        def update_range(self, l, r, val):
            # Update the range [l, r) with the minimum of current and val
            # Here, l and r are 0-based
            l += self.n
            r += self.n
            while l < r:
                if l % 2 == 1:
                    if self.tree[l] > val:
                        self.tree[l] = val
                    l += 1
                if r % 2 == 1:
                    r -= 1
                    if self.tree[r] > val:
                        self.tree[r] = val
                l >>=1
                r >>=1

        def query_range(self, l, r):
            # Find the minimum in [l, r)
            res = self.default
            l += self.n
            r += self.n
            while l < r:
                if l % 2 == 1:
                    res = min(res, self.tree[l])
                    l +=1
                if r % 2 ==1:
                    r -=1
                    res = min(res, self.tree[r])
                l >>=1
                r >>=1
            return res
        
        def get(self, pos):
            # Get the value at a specific position
            pos += self.n
            res = self.default
            while pos >=1:
                res = min(res, self.tree[pos])
                pos >>=1
            return res

    # Initialize the segment tree with H as the initial blocked row (since rows are 0-based, the bottom is H-1)
    st = SegmentTree(W, H)
    res = []

    # Process the bars in reverse order
    for bar in reversed(bars):
        R, C, L = bar
        # The columns are C to C+L-1, inclusive
        end_col = C + L -1
        min_floor = st.query_range(C, end_col+1)
        # The bar can move down to min_floor -1, but can't go beyond H-1
        final_row = min(min_floor -1, H-1)
        # The bar's final row is the maximum between its initial row and this value
        final_row = max(R, final_row)
        res.append(final_row +1)  # convert back to 1-based
        # Update the segment tree for the columns C to end_col to have the new floor
        st.update_range(C, end_col+1, final_row)

    # Reverse the results to get the correct order
    res = res[::-1]
    for r in res:
        print(r)

if __name__ == '__main__':
    main()