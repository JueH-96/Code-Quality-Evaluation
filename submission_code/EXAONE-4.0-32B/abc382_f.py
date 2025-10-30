import sys
sys.setrecursionlimit(3000000)
INF = 10**18

class SegmentTree:
    def __init__(self, n):
        self.n = n
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.tree = [INF] * (2 * self.size)
        self.lazy = [INF] * (2 * self.size)
    
    def push(self, node):
        if self.lazy[node] == INF:
            return
        if node < self.size:
            self.tree[2*node] = min(self.tree[2*node], self.lazy[node])
            self.lazy[2*node] = min(self.lazy[2*node], self.lazy[node])
            self.tree[2*node+1] = min(self.tree[2*node+1], self.lazy[node])
            self.lazy[2*node+1] = min(self.lazy[2*node+1], self.lazy[node])
        self.lazy[node] = INF

    def update(self, l, r, value, node=1, segl=1, segr=None):
        if segr is None:
            segr = self.size
        if r < segl or l > segr:
            return
        if l <= segl and segr <= r:
            self.tree[node] = min(self.tree[node], value)
            self.lazy[node] = min(self.lazy[node], value)
            return
        self.push(node)
        mid = (segl + segr) // 2
        self.update(l, r, value, 2*node, segl, mid)
        self.update(l, r, value, 2*node+1, mid+1, segr)
        self.tree[node] = min(self.tree[2*node], self.tree[2*node+1])

    def query(self, l, r, node=1, segl=1, segr=None):
        if segr is None:
            segr = self.size
        if r < segl or l > segr:
            return INF
        if l <= segl and segr <= r:
            return self.tree[node]
        self.push(node)
        mid = (segl + segr) // 2
        left_val = self.query(l, r, 2*node, segl, mid)
        right_val = self.query(l, r, 2*node+1, mid+1, segr)
        return min(left_val, right_val)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    H = int(data[0]); W = int(data[1]); N = int(data[2])
    bars = []
    idx = 3
    for i in range(N):
        R = int(data[idx]); C = int(data[idx+1]); L = int(data[idx+2]); idx += 3
        bars.append((R, C, L, i))
    bars.sort(key=lambda x: (-x[0], x[3]))
    seg_tree = SegmentTree(W)
    res = [0] * N

    for bar in bars:
        R, C, L, orig_idx = bar
        end = C + L - 1
        min_obstacle = seg_tree.query(C, end)
        if min_obstacle == INF:
            final_row = H
        else:
            final_row = min_obstacle - 1
        res[orig_idx] = final_row
        seg_tree.update(C, end, final_row)

    for i in range(N):
        print(res[i])

if __name__ == '__main__':
    main()