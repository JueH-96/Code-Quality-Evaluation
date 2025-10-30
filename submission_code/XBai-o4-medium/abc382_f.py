import sys

class SegmentTree:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = size
        self.inf = float('inf')
        self.tree = [self.inf] * (2 * self.n)
        self.lazy = [self.inf] * (2 * self.n)
    
    def push(self, node, l, r):
        if self.lazy[node] == self.inf:
            return
        if l != r:
            # Propagate to children
            self.tree[2*node] = min(self.tree[2*node], self.lazy[node])
            self.tree[2*node+1] = min(self.tree[2*node+1], self.lazy[node])
            self.lazy[2*node] = min(self.lazy[2*node], self.lazy[node])
            self.lazy[2*node+1] = min(self.lazy[2*node+1], self.lazy[node])
        self.lazy[node] = self.inf
    
    def update_range(self, a, b, v, node=1, l=0, r=None):
        if r is None:
            r = self.n - 1
        self.push(node, l, r)
        if a > r or b < l:
            return
        if a <= l and r <= b:
            self.tree[node] = min(self.tree[node], v)
            if l != r:
                self.lazy[node] = min(self.lazy[node], v)
            return
        mid = (l + r) // 2
        self.update_range(a, b, v, 2*node, l, mid)
        self.update_range(a, b, v, 2*node+1, mid+1, r)
        self.tree[node] = min(self.tree[2*node], self.tree[2*node+1])
    
    def query_range(self, a, b, node=1, l=0, r=None):
        if r is None:
            r = self.n - 1
        self.push(node, l, r)
        if a > r or b < l:
            return self.inf
        if a <= l and r <= b:
            return self.tree[node]
        mid = (l + r) // 2
        left = self.query_range(a, b, 2*node, l, mid)
        right = self.query_range(a, b, 2*node+1, mid+1, r)
        return min(left, right)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr]); ptr +=1
    W = int(input[ptr]); ptr +=1
    N = int(input[ptr]); ptr +=1
    
    bars = []
    for _ in range(N):
        R = int(input[ptr]); ptr +=1
        C = int(input[ptr]); ptr +=1
        L = int(input[ptr]); ptr +=1
        bars.append( (R, C, L) )
    
    # Sort bars in descending order of R, then any order
    sorted_bars = sorted( [ (i, bar) for i, bar in enumerate(bars) ], key=lambda x: (-x[1][0], ) )
    
    # Initialize segment tree
    st = SegmentTree(W)
    result = [0] * N
    
    for orig_idx, (R, C, L) in sorted_bars:
        c_start = C - 1
        c_end = C - 1 + L - 1
        # Query [c_start, c_end]
        m = st.query_range(c_start, c_end)
        if m == st.inf:
            final_row = H
        else:
            final_row = m - 1
        result[orig_idx] = final_row
        # Update [c_start, c_end] with final_row
        st.update_range(c_start, c_end, final_row)
    
    for res in result:
        print(res)

if __name__ == '__main__':
    main()