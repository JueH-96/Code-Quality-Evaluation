def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    Q = int(next(it))
    
    # We'll use a Fenwick tree (Binary Indexed Tree) to maintain dynamically
    # which cells (walls) are still present in each row and in each column.
    # We can then quickly query, for a given row, the "first" wall to the left or
    # right of a given column and similarly for columns.
    #
    # Fenw trees are implemented 1-indexed.
    
    class Fenw:
        __slots__ = ('n','tree')
        def __init__(self, n):
            self.n = n
            self.tree = [0] * (n+1)
        def build(self, arr):
            # Given a 0-indexed array of length n, build the Fenw tree.
            for i in range(1, self.n+1):
                self.tree[i] = arr[i-1]
            i = 1
            while i <= self.n:
                j = i + (i & -i)
                if j <= self.n:
                    self.tree[j] += self.tree[i]
                i += 1
        def update(self, i, delta):
            # Update index i (1-indexed) by adding delta.
            while i <= self.n:
                self.tree[i] += delta
                i += i & -i
        def query(self, i):
            # Sum from 1 to i.
            s = 0
            while i:
                s += self.tree[i]
                i -= i & -i
            return s
        def search(self, k):
            # Find the smallest index i such that query(i) >= k.
            # k must be >= 1 and such index must exist.
            idx = 0
            bit = 1 << (self.n.bit_length()-1)
            while bit:
                nxt = idx + bit
                if nxt <= self.n and self.tree[nxt] < k:
                    k -= self.tree[nxt]
                    idx = nxt
                bit //= 2
            return idx + 1

    # Create a Fenw tree for each row.
    # For row r (1-indexed), we have W columns (indices 1..W) each initially 1 (wall exists).
    rowFenw = [None] * (H+1)
    # Total number of fenw elements (summed over rows) is H*W which is at most 400,000.
    for r in range(1, H+1):
        fenw = Fenw(W)
        fenw.build([1] * W)
        rowFenw[r] = fenw

    # Similarly, create a Fenw tree for each column.
    colFenw = [None] * (W+1)
    for c in range(1, W+1):
        fenw = Fenw(H)
        fenw.build([1] * H)
        colFenw[c] = fenw

    # We'll also track the grid state using a 2D boolean array.
    # grid[r][c] = True means a wall is present at (r,c)
    # We use 1-indexing for both rows and columns.
    grid = [None] * (H+1)
    for r in range(H+1):
        grid[r] = [False] * (W+1)
    for r in range(1, H+1):
        for c in range(1, W+1):
            grid[r][c] = True

    # Global count of walls.
    remaining = H * W

    # Helper function that “removes” a wall (if it exists) from the grid 
    # and updates both fenw data structures.
    def remove_cell(r, c):
        nonlocal remaining
        if r < 1 or r > H or c < 1 or c > W:
            return
        if grid[r][c]:
            grid[r][c] = False
            rowFenw[r].update(c, -1)
            colFenw[c].update(r, -1)
            remaining -= 1

    # Process each query in order.
    # For each bomb at (r, c):
    #   - If a wall exists at (r, c) then destroy that wall.
    #   - Otherwise (if already empty), then in each of the four directions,
    #     find and destroy the first wall (if one exists).
    for _ in range(Q):
        r = int(next(it))
        c = int(next(it))
        if grid[r][c]:
            # Bomb is placed on a cell with a wall, so just remove that cell.
            remove_cell(r, c)
        else:
            # Bomb is placed on an empty cell.
            # For the vertical direction:
            # Check upward (same column, row < r).
            if r > 1:
                # Query the count of walls in column c among rows 1..(r-1)
                cnt = colFenw[c].query(r-1)
                if cnt > 0:
                    # The cnt-th wall in col c (when counted from top) is the first wall when looking upward.
                    up_i = colFenw[c].search(cnt)
                    remove_cell(up_i, c)
            # Check downward (same column, row > r).
            if r < H:
                cnt_here = colFenw[c].query(r)
                total = colFenw[c].query(H)
                if total - cnt_here > 0:
                    down_i = colFenw[c].search(cnt_here + 1)
                    remove_cell(down_i, c)
            # For the horizontal direction:
            # Check left (same row, col < c).
            if c > 1:
                cnt = rowFenw[r].query(c-1)
                if cnt > 0:
                    left_j = rowFenw[r].search(cnt)
                    remove_cell(r, left_j)
            # Check right (same row, col > c).
            if c < W:
                cnt_here = rowFenw[r].query(c)
                total = rowFenw[r].query(W)
                if total - cnt_here > 0:
                    right_j = rowFenw[r].search(cnt_here + 1)
                    remove_cell(r, right_j)
    sys.stdout.write(str(remaining))

if __name__ == '__main__':
    main()