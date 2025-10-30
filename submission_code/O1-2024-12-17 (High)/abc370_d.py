def main():
    import sys
    input_data = sys.stdin.read().split()
    # Fast parsing of input
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    queries = input_data[3:]
    
    # A Fenwick (Binary Indexed) Tree class to keep track of "existing walls" in 1D
    class Fenwick:
        __slots__ = ['n', 'fw']
        def __init__(self, n):
            self.n = n
            self.fw = [0]*(n+1)
        def build_all_ones(self):
            # Build the Fenwick tree as if the array = [1,1,...,1] of length n
            for i in range(1, self.n+1):
                self.fw[i] = 1
            # O(n) Fenwick build
            for i in range(1, self.n+1):
                j = i + (i & -i)
                if j <= self.n:
                    self.fw[j] += self.fw[i]
        def update(self, idx, delta):
            # idx in [0..n-1], do fenwicks at (idx+1)
            i = idx + 1
            while i <= self.n:
                self.fw[i] += delta
                i += i & -i
        def query(self, idx):
            # returns sum in range [0..idx], inclusive, 0-based
            if idx < 0:
                return 0
            s = 0
            i = idx + 1
            while i > 0:
                s += self.fw[i]
                i -= i & -i
            return s
        def get_index_of_kth_one(self, k):
            # Find the 1-based Fenwick index where the cumulative sum == k
            # (i.e. the position of the k-th "1" in 0-based sense).
            # Returns a 1-based index.  Caller typically does "-1" for 0-based.
            idx = 0
            bit_mask = 1 << (self.n.bit_length() - 1)
            while bit_mask > 0:
                nxt = idx + bit_mask
                if nxt <= self.n and self.fw[nxt] < k:
                    k -= self.fw[nxt]
                    idx = nxt
                bit_mask >>= 1
            return idx
    
    # We will store "walls" in a single 1D array of length H*W
    # walls[r*W + c] = True/False
    walls = [True]*(H*W)
    def has_wall(r, c):
        return walls[r*W + c]
    def destroy_wall(r, c):
        # Mark the wall as destroyed and update fenwicks
        walls[r*W + c] = False
        nonlocal total_walls
        total_walls -= 1
        rowFen[r].update(c, -1)
        colFen[c].update(r, -1)
    
    # Fenwicks for each row (to manage columns) and for each column (to manage rows)
    rowFen = [Fenwick(W) for _ in range(H)]
    colFen = [Fenwick(H) for _ in range(W)]
    
    # Build all-ones Fenwicks in O(H*W)
    for r in range(H):
        rowFen[r].build_all_ones()
    for c in range(W):
        colFen[c].build_all_ones()
    
    total_walls = H*W
    
    # Helpers to find the "first wall" in each direction if the cell itself is empty:
    def find_wall_above(r, c):
        if r <= 0:
            return None
        fw = colFen[c]
        # Number of walls from row 0 up to r-1
        s = fw.query(r-1)
        if s == 0:
            return None
        # The largest row index < r that is still a wall is the s-th wall
        row_above = fw.get_index_of_kth_one(s) - 1  # convert to 0-based
        return row_above if row_above >= 0 else None
    
    def find_wall_below(r, c):
        if r >= H-1:
            return None
        fw = colFen[c]
        # s1 = number of walls up to and including r
        s1 = fw.query(r)
        total = fw.query(H-1)
        if s1 == total:  # no wall below
            return None
        row_below = fw.get_index_of_kth_one(s1+1) - 1
        # must be strictly greater than r
        if row_below <= r:
            return None
        return row_below
    
    def find_wall_left(r, c):
        if c <= 0:
            return None
        fw = rowFen[r]
        s = fw.query(c-1)
        if s == 0:
            return None
        col_left = fw.get_index_of_kth_one(s) - 1
        return col_left if col_left >= 0 else None
    
    def find_wall_right(r, c):
        if c >= W-1:
            return None
        fw = rowFen[r]
        s1 = fw.query(c)
        total = fw.query(W-1)
        if s1 == total:  # no wall to the right
            return None
        col_right = fw.get_index_of_kth_one(s1+1) - 1
        if col_right <= c:
            return None
        return col_right
    
    # Process queries
    idx = 0
    for _ in range(Q):
        rr = int(queries[idx]); cc = int(queries[idx+1])
        idx += 2
        r = rr - 1
        c = cc - 1
        
        if has_wall(r, c):
            # Destroy the wall in the bomb cell and finish
            destroy_wall(r, c)
        else:
            # Destroy the first wall in each direction (simultaneously)
            up = find_wall_above(r, c)
            down = find_wall_below(r, c)
            left = find_wall_left(r, c)
            right = find_wall_right(r, c)
            
            # Collect them, then destroy
            # (Simultaneous => we find all first, then destroy)
            targets = []
            if up is not None:
                targets.append((up, c))
            if down is not None:
                targets.append((down, c))
            if left is not None:
                targets.append((r, left))
            if right is not None:
                targets.append((r, right))
            # Destroy them
            for (rrr, ccc) in targets:
                if has_wall(rrr, ccc):
                    destroy_wall(rrr, ccc)
    
    # Output the result: number of walls remaining
    print(total_walls)

# Do not forget to call main()
main()