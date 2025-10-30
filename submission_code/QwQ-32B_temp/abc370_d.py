import sys

class SegmentTree:
    def __init__(self, size):
        self.n = size
        self.size_tree = 1
        while self.size_tree < self.n:
            self.size_tree <<= 1
        self.max_tree = [-float('inf')] * (2 * self.size_tree)
        self.min_tree = [float('inf')] * (2 * self.size_tree)
    
    def build(self, data):
        for i in range(self.n):
            self.max_tree[self.size_tree + i] = data[i]
            self.min_tree[self.size_tree + i] = data[i]
        for i in range(self.size_tree - 1, 0, -1):
            self.max_tree[i] = max(self.max_tree[2*i], self.max_tree[2*i+1])
            self.min_tree[i] = min(self.min_tree[2*i], self.min_tree[2*i+1])
    
    def update(self, pos, val):
        pos += self.size_tree
        self.max_tree[pos] = val
        self.min_tree[pos] = val
        pos >>= 1
        while pos >= 1:
            new_max = max(self.max_tree[2*pos], self.max_tree[2*pos+1])
            new_min = min(self.min_tree[2*pos], self.min_tree[2*pos+1])
            if self.max_tree[pos] == new_max and self.min_tree[pos] == new_min:
                break
            self.max_tree[pos] = new_max
            self.min_tree[pos] = new_min
            pos >>= 1
    
    def query_max(self, l, r):
        res = -float('inf')
        l += self.size_tree
        r += self.size_tree
        while l < r:
            if l % 2 == 1:
                res = max(res, self.max_tree[l])
                l += 1
            if r % 2 == 1:
                r -= 1
                res = max(res, self.max_tree[r])
            l >>= 1
            r >>= 1
        return res
    
    def query_min(self, l, r):
        res = float('inf')
        l += self.size_tree
        r += self.size_tree
        while l < r:
            if l % 2 == 1:
                res = min(res, self.min_tree[l])
                l += 1
            if r % 2 == 1:
                r -= 1
                res = min(res, self.min_tree[r])
            l >>= 1
            r >>= 1
        return res

def main():
    H, W, Q = map(int, sys.stdin.readline().split())
    present = [[False]*(W+1) for _ in range(H+1)]
    for r in range(1, H+1):
        for c in range(1, W+1):
            present[r][c] = True
    
    rows_st = []
    for r in range(1, H+1):
        data = []
        for c in range(1, W+1):
            if present[r][c]:
                data.append(c)
            else:
                data.append(-float('inf'))
        st = SegmentTree(W)
        st.build(data)
        rows_st.append(st)
    
    cols_st = []
    for c in range(1, W+1):
        data = []
        for r in range(1, H+1):
            if present[r][c]:
                data.append(r)
            else:
                data.append(-float('inf'))
        st = SegmentTree(H)
        st.build(data)
        cols_st.append(st)
    
    for _ in range(Q):
        R, C = map(int, sys.stdin.readline().split())
        if present[R][C]:
            present[R][C] = False
            pos = C - 1
            rows_st[R-1].update(pos, -float('inf'))
            pos_col = R - 1
            cols_st[C-1].update(pos_col, -float('inf'))
        else:
            # Up direction (column C, rows < R)
            st_col = cols_st[C-1]
            max_row = st_col.query_max(0, R-1)
            if max_row != -float('inf'):
                row_i = max_row
                present[row_i][C] = False
                pos_row = C - 1
                rows_st[row_i-1].update(pos_row, -float('inf'))
                pos_col = row_i - 1
                cols_st[C-1].update(pos_col, -float('inf'))
            # Down direction (rows > R)
            min_row = st_col.query_min(R, H)
            if min_row != float('inf'):
                row_i = min_row
                present[row_i][C] = False
                pos_row = C - 1
                rows_st[row_i-1].update(pos_row, -float('inf'))
                pos_col = row_i - 1
                cols_st[C-1].update(pos_col, -float('inf'))
            # Left direction (columns < C)
            st_row = rows_st[R-1]
            max_col = st_row.query_max(0, C-1)
            if max_col != -float('inf'):
                col_i = max_col
                present[R][col_i] = False
                pos_row = col_i - 1
                rows_st[R-1].update(pos_row, -float('inf'))
                pos_col = R - 1
                cols_st[col_i-1].update(pos_col, -float('inf'))
            # Right direction (columns > C)
            min_col = st_row.query_min(C, W)
            if min_col != float('inf'):
                col_i = min_col
                present[R][col_i] = False
                pos_row = col_i - 1
                rows_st[R-1].update(pos_row, -float('inf'))
                pos_col = R - 1
                cols_st[col_i-1].update(pos_col, -float('inf'))
    
    total = 0
    for r in range(1, H+1):
        for c in range(1, W+1):
            if present[r][c]:
                total += 1
    print(total)

if __name__ == "__main__":
    main()