class SegmentTree:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.left = None
        self.right = None
        if start == end:
            self.max_val = start
            self.min_val = start
        else:
            mid = (start + end) // 2
            self.left = SegmentTree(start, mid)
            self.right = SegmentTree(mid + 1, end)
            self.max_val = max(self.left.max_val, self.right.max_val)
            self.min_val = min(self.left.min_val, self.right.min_val)
    
    def update(self, pos):
        if self.start == self.end:
            self.max_val = -float('inf')
            self.min_val = float('inf')
            return
        if pos <= self.left.end:
            self.left.update(pos)
        else:
            self.right.update(pos)
        self.max_val = max(self.left.max_val, self.right.max_val)
        self.min_val = min(self.left.min_val, self.right.min_val)
    
    def query_max(self, l, r):
        if r < self.start or l > self.end:
            return -float('inf')
        if l <= self.start and self.end <= r:
            return self.max_val
        return max(self.left.query_max(l, r), self.right.query_max(l, r))
    
    def query_min(self, l, r):
        if r < self.start or l > self.end:
            return float('inf')
        if l <= self.start and self.end <= r:
            return self.min_val
        return min(self.left.query_min(l, r), self.right.query_min(l, r))

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    rows = {}
    for r in range(1, H+1):
        rows[r] = SegmentTree(1, W)
    
    cols = {}
    for c in range(1, W+1):
        cols[c] = SegmentTree(1, H)
    
    total_walls = H * W
    
    for _ in range(Q):
        R = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        
        row_tree = rows[R]
        max_col = row_tree.query_max(C, C)
        if max_col == C:
            row_tree.update(C)
            col_tree = cols[C]
            col_tree.update(R)
            total_walls -= 1
        else:
            walls = []
            col_tree = cols[C]
            max_row = col_tree.query_max(1, R-1)
            if max_row != -float('inf'):
                walls.append((max_row, C))
            
            min_row = col_tree.query_min(R+1, H)
            if min_row != float('inf'):
                walls.append((min_row, C))
            
            row_tree = rows[R]
            max_col = row_tree.query_max(1, C-1)
            if max_col != -float('inf'):
                walls.append((R, max_col))
            
            min_col = row_tree.query_min(C+1, W)
            if min_col != float('inf'):
                walls.append((R, min_col))
            
            for r, c in walls:
                rows[r].update(c)
                cols[c].update(r)
                total_walls -= 1
    
    print(total_walls)

if __name__ == '__main__':
    main()