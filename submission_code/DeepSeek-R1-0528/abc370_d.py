import sys

INF = 10**9
NEG_INF = -10**9

class SegmentTreeMax:
    def __init__(self, n, initial_values):
        self.n = n
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.data = [NEG_INF] * (2 * self.size)
        for i in range(n):
            self.data[self.size + i] = initial_values[i]
        for i in range(self.size-1, 0, -1):
            self.data[i] = max(self.data[2*i], self.data[2*i+1])
    
    def update(self, index, value):
        pos = index - 1
        if pos < 0 or pos >= self.n:
            return
        idx = self.size + pos
        self.data[idx] = value
        idx //= 2
        while idx:
            self.data[idx] = max(self.data[2*idx], self.data[2*idx+1])
            idx //= 2

    def query(self, l, r):
        if l > r:
            return NEG_INF
        l0 = l - 1
        r0 = r - 1
        if l0 < 0 or r0 >= self.n or l0 > r0:
            return NEG_INF
        l0 += self.size
        r0 += self.size
        res = NEG_INF
        while l0 <= r0:
            if l0 % 2 == 1:
                res = max(res, self.data[l0])
                l0 += 1
            if r0 % 2 == 0:
                res = max(res, self.data[r0])
                r0 -= 1
            l0 //= 2
            r0 //= 2
        return res

class SegmentTreeMin:
    def __init__(self, n, initial_values):
        self.n = n
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.data = [INF] * (2 * self.size)
        for i in range(n):
            self.data[self.size + i] = initial_values[i]
        for i in range(self.size-1, 0, -1):
            self.data[i] = min(self.data[2*i], self.data[2*i+1])
    
    def update(self, index, value):
        pos = index - 1
        if pos < 0 or pos >= self.n:
            return
        idx = self.size + pos
        self.data[idx] = value
        idx //= 2
        while idx:
            self.data[idx] = min(self.data[2*idx], self.data[2*idx+1])
            idx //= 2

    def query(self, l, r):
        if l > r:
            return INF
        l0 = l - 1
        r0 = r - 1
        if l0 < 0 or r0 >= self.n or l0 > r0:
            return INF
        l0 += self.size
        r0 += self.size
        res = INF
        while l0 <= r0:
            if l0 % 2 == 1:
                res = min(res, self.data[l0])
                l0 += 1
            if r0 % 2 == 0:
                res = min(res, self.data[r0])
                r0 -= 1
            l0 //= 2
            r0 //= 2
        return res

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    H = int(data[0])
    W = int(data[1])
    Q = int(data[2])
    queries = []
    idx = 3
    for i in range(Q):
        r = int(data[idx])
        c = int(data[idx+1])
        idx += 2
        queries.append((r, c))
    
    total_walls = H * W
    destroyed = 0

    row_max_trees = [None] * (H+1)
    row_min_trees = [None] * (H+1)
    for i in range(1, H+1):
        arr = list(range(1, W+1))
        row_max_trees[i] = SegmentTreeMax(W, arr)
        row_min_trees[i] = SegmentTreeMin(W, arr)
    
    col_max_trees = [None] * (W+1)
    col_min_trees = [None] * (W+1)
    for j in range(1, W+1):
        arr = list(range(1, H+1))
        col_max_trees[j] = SegmentTreeMax(H, arr)
        col_min_trees[j] = SegmentTreeMin(H, arr)
    
    for (r, c) in queries:
        if row_max_trees[r].query(c, c) == NEG_INF:
            walls_to_destroy = []
            left_val = row_max_trees[r].query(1, c-1)
            if left_val != NEG_INF:
                walls_to_destroy.append((r, left_val))
            
            right_val = row_min_trees[r].query(c+1, W)
            if right_val != INF:
                walls_to_destroy.append((r, right_val))
            
            above_val = col_max_trees[c].query(1, r-1)
            if above_val != NEG_INF:
                walls_to_destroy.append((above_val, c))
            
            below_val = col_min_trees[c].query(r+1, H)
            if below_val != INF:
                walls_to_destroy.append((below_val, c))
            
            for (x, y) in walls_to_destroy:
                if 1 <= x <= H and 1 <= y <= W:
                    row_max_trees[x].update(y, NEG_INF)
                    row_min_trees[x].update(y, INF)
                if 1 <= y <= W and 1 <= x <= H:
                    col_max_trees[y].update(x, NEG_INF)
                    col_min_trees[y].update(x, INF)
            
            destroyed += len(walls_to_destroy)
        else:
            row_max_trees[r].update(c, NEG_INF)
            row_min_trees[r].update(c, INF)
            col_max_trees[c].update(r, NEG_INF)
            col_min_trees[c].update(r, INF)
            destroyed += 1

    print(total_walls - destroyed)

if __name__ == '__main__':
    main()