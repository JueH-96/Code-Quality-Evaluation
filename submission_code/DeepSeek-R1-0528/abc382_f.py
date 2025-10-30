import sys

class SegmentTreeMin:
    def __init__(self, n, init_val):
        self.n = n
        self.INF = 10**18
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.tree = [self.INF] * (2 * self.size)
        self.lazy = [self.INF] * (2 * self.size)
        for i in range(n):
            self.tree[self.size + i] = init_val
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])
    
    def apply(self, idx, val):
        self.tree[idx] = min(self.tree[idx], val)
        if idx < self.size:
            self.lazy[idx] = min(self.lazy[idx], val)
    
    def push(self, idx):
        if self.lazy[idx] == self.INF:
            return
        self.apply(2*idx, self.lazy[idx])
        self.apply(2*idx+1, self.lazy[idx])
        self.lazy[idx] = self.INF

    def update_range(self, l, r, val):
        stack = [(1, 0, self.size-1)]
        updates = []
        while stack:
            idx, segL, segR = stack.pop()
            if r < segL or l > segR:
                continue
            if l <= segL and segR <= r:
                self.apply(idx, val)
                continue
            self.push(idx)
            mid = (segL + segR) // 2
            stack.append((2*idx+1, mid+1, segR))
            stack.append((2*idx, segL, mid))
        self.rebuild_path(l, r, 1, 0, self.size-1)
        
    def rebuild_path(self, l, r, idx, segL, segR):
        if segL == segR:
            return
        if r < segL or l > segR:
            return
        mid = (segL + segR) // 2
        self.push(idx)
        self.rebuild_path(l, r, 2*idx, segL, mid)
        self.rebuild_path(l, r, 2*idx+1, mid+1, segR)
        self.tree[idx] = min(self.tree[2*idx], self.tree[2*idx+1])
    
    def query_range(self, l, r):
        stack = [(1, 0, self.size-1)]
        result = self.INF
        while stack:
            idx, segL, segR = stack.pop()
            if r < segL or l > segR:
                continue
            if l <= segL and segR <= r:
                result = min(result, self.tree[idx])
                continue
            self.push(idx)
            mid = (segL + segR) // 2
            stack.append((2*idx+1, mid+1, segR))
            stack.append((2*idx, segL, mid))
        return result

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    H = int(data[0]); W = int(data[1]); N = int(data[2])
    bars = []
    index_ptr = 3
    for i in range(N):
        r = int(data[index_ptr]); c = int(data[index_ptr+1]); l_val = int(data[index_ptr+2])
        index_ptr += 3
        bars.append((r, c, l_val, i))
    
    bars_sorted = sorted(bars, key=lambda x: (-x[0], x[3]))
    
    seg_tree = SegmentTreeMin(W, H+1)
    res = [0] * N

    for bar in bars_sorted:
        r, c, l_val, orig_idx = bar
        left_col = c - 1
        right_col = c + l_val - 2
        obstacle = seg_tree.query_range(left_col, right_col)
        stop_row = obstacle - 1
        res[orig_idx] = stop_row
        seg_tree.update_range(left_col, right_col, stop_row)
    
    for i in range(N):
        print(res[i])

if __name__ == "__main__":
    main()