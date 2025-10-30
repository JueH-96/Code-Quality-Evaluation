class SegmentTree:
    def __init__(self, size, initial_value):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = size
        self.tree = [initial_value] * (2 * self.n)
        self.lazy = [float('inf')] * (2 * self.n)
    
    def apply(self, node, l, r, val):
        self.tree[node] = min(self.tree[node], val)
        if l != r:
            self.lazy[node] = min(self.lazy[node], val)
    
    def push(self, node, l, r):
        if self.lazy[node] != float('inf'):
            mid = (l + r) // 2
            left = 2 * node
            right_node = 2 * node + 1
            self.apply(left, l, mid, self.lazy[node])
            self.apply(right_node, mid + 1, r, self.lazy[node])
            self.lazy[node] = float('inf')
    
    def range_min_update(self, a, b, val):
        def update(node, l, r):
            if a > r or b < l:
                return
            if a <= l and r <= b:
                self.apply(node, l, r, val)
                return
            self.push(node, l, r)
            mid = (l + r) // 2
            update(2 * node, l, mid)
            update(2 * node + 1, mid + 1, r)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])
        update(1, 0, self.n - 1)
    
    def query_min(self, a, b):
        def query(node, l, r):
            if a > r or b < l:
                return float('inf')
            if a <= l and r <= b:
                return self.tree[node]
            self.push(node, l, r)
            mid = (l + r) // 2
            left_val = query(2 * node, l, mid)
            right_val = query(2 * node + 1, mid + 1, r)
            return min(left_val, right_val)
        return query(1, 0, self.n - 1)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    W = int(data[idx+1])
    N = int(data[idx+2])
    idx +=3
    
    bars = []
    for i in range(N):
        R = int(data[idx])
        C = int(data[idx+1])
        L = int(data[idx+2])
        bars.append((R, C, L, i))
        idx +=3
    
    sorted_bars = sorted(bars, key=lambda x: (-x[0], -x[3]))
    ans = [0] * N
    initial_val = H + 1
    if W == 0:
        for bar in bars:
            ans[bar[3]] = bar[0]
        for num in ans:
            print(num)
        return
    st = SegmentTree(W, initial_val)
    
    for bar in sorted_bars:
        R, C, L, original_idx = bar
        left = C - 1
        right = C + L - 2
        current_min = st.query_min(left, right)
        if current_min == initial_val:
            candidate = H
        else:
            candidate = current_min - 1
        R_prime = max(R, candidate)
        R_prime = min(R_prime, H)
        ans[original_idx] = R_prime
        st.range_min_update(left, right, R_prime)
    
    for num in ans:
        print(num)

if __name__ == '__main__':
    main()