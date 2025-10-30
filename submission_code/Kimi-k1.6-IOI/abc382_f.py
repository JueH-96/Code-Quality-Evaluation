import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1
    N = int(input[ptr])
    ptr += 1

    bars = []
    for i in range(N):
        R = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        c_start = C - 1
        c_end = c_start + L - 1
        bars.append((-R, R, c_start, c_end, i))
    
    bars.sort()

    class SegmentTree:
        def __init__(self, size):
            self.n = 1
            while self.n < size:
                self.n <<= 1
            self.size = size
            self.tree = [float('inf')] * (2 * self.n)
            self.lazy = [float('inf')] * (2 * self.n)
        
        def push(self, node, l, r):
            if self.lazy[node] == float('inf'):
                return
            self.tree[node] = min(self.tree[node], self.lazy[node])
            if l != r:
                self.lazy[2*node] = min(self.lazy[2*node], self.lazy[node])
                self.lazy[2*node+1] = min(self.lazy[2*node+1], self.lazy[node])
            self.lazy[node] = float('inf')
        
        def update_range_min(self, a, b, val):
            def update(node, l, r):
                self.push(node, l, r)
                if a > r or b < l:
                    return
                if a <= l and r <= b:
                    self.lazy[node] = min(self.lazy[node], val)
                    self.push(node, l, r)
                    return
                mid = (l + r) // 2
                update(2*node, l, mid)
                update(2*node+1, mid+1, r)
                self.tree[node] = min(self.tree[2*node], self.tree[2*node+1])
            update(1, 0, self.n - 1)
        
        def query_min(self, a, b):
            def query(node, l, r):
                self.push(node, l, r)
                if a > r or b < l:
                    return float('inf')
                if a <= l and r <= b:
                    return self.tree[node]
                mid = (l + r) // 2
                left = query(2*node, l, mid)
                right = query(2*node+1, mid+1, r)
                return min(left, right)
            return query(1, 0, self.n - 1)
    
    st = SegmentTree(W)
    result = [0] * N

    for bar in bars:
        _, R_i, c_start, c_end, idx = bar
        min_row = st.query_min(c_start, c_end)
        if min_row == float('inf'):
            R_prime = H
        else:
            R_prime = min(H, min_row - 1)
        result[idx] = R_prime
        st.update_range_min(c_start, c_end, R_prime)
    
    for r in result:
        print(r)

if __name__ == '__main__':
    main()