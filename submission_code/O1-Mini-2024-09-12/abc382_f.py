# YOUR CODE HERE
import sys
import sys
import sys
import sys
import sys
sys.setrecursionlimit(1 << 25)

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    H, W, N = map(int, data[:3])
    bars = []
    idx = 3
    for i in range(1, N+1):
        R_i = int(data[idx])
        C_i = int(data[idx+1])
        L_i = int(data[idx+2])
        bars.append( ( -R_i, i, C_i, L_i ) )  # sort by decreasing R_i, increasing i
        idx +=3
    bars_sorted = sorted(bars)
    
    # Implement Segment Tree
    class SegmentTree:
        __slots__ = ['n', 'size', 'minv', 'lazy']

        def __init__(self, size):
            self.n = 1
            while self.n < size:
                self.n <<=1
            self.size = self.n <<1
            self.minv = [H+1] * (self.n <<1)
            self.lazy = [None] * (self.n <<1)
        
        def push(self, node, l, r):
            if self.lazy[node] is not None:
                self.minv[node] = self.lazy[node]
                if node < self.n:
                    self.lazy[node<<1] = self.lazy[node]
                    self.lazy[node<<1|1] = self.lazy[node]
                self.lazy[node] = None
        
        def range_min(self, L, R):
            def _range_min(node, l, r):
                self.push(node, l, r)
                if R < l or r < L:
                    return H+1
                if L <= l and r <= R:
                    return self.minv[node]
                mid = (l + r) >>1
                return min(_range_min(node<<1, l, mid), _range_min(node<<1|1, mid+1, r))
            return _range_min(1, 1, self.n)
        
        def range_assign(self, L, R, val):
            def _range_assign(node, l, r):
                self.push(node, l, r)
                if R < l or r < L:
                    return
                if L <= l and r <= R:
                    self.lazy[node] = val
                    self.push(node, l, r)
                    return
                mid = (l + r) >>1
                _range_assign(node<<1, l, mid)
                _range_assign(node<<1|1, mid+1, r)
                self.minv[node] = min(self.minv[node<<1], self.minv[node<<1|1])
            _range_assign(1,1,self.n)
            # After lazy assign, set range to val
            def _assign(node, l, r):
                self.push(node, l, r)
                if R < l or r < L:
                    return
                if L <= l and r <= R:
                    self.minv[node] = val
                    if node < self.n:
                        self.lazy[node<<1] = val
                        self.lazy[node<<1|1] = val
                    return
                mid = (l + r) >>1
                _assign(node<<1, l, mid)
                _assign(node<<1|1, mid+1, r)
                self.minv[node] = min(self.minv[node<<1], self.minv[node<<1|1])
            _assign(1,1,self.n)
        
        def query_min_fast(self, L, R):
            res = H+1
            L += self.n -1
            R += self.n -1
            while L <= R:
                if L %2 ==1:
                    if self.lazy_nodes[L] is not None:
                        res = min(res, self.minv[L])
                    else:
                        res = min(res, self.minv[L])
                    L +=1
                if R %2 ==0:
                    if self.lazy_nodes[R] is not None:
                        res = min(res, self.minv[R])
                    else:
                        res = min(res, self.minv[R])
                    R -=1
                L >>=1
                R >>=1
            return res

    # Implement a faster segment tree with push and assign
    class FastSegmentTree:
        def __init__(self, size):
            self.N = 1
            while self.N < size:
                self.N <<=1
            self.size = self.N <<1
            self.minv = [H+1] * (self.N <<1)
            self.lazy = [None] * (self.N <<1)
        
        def push_down(self, node, l, r):
            if self.lazy[node] is not None:
                self.minv[node] = self.lazy[node]
                if node < self.N:
                    self.lazy[node<<1] = self.lazy[node]
                    self.lazy[node<<1|1] = self.lazy[node]
                self.lazy[node] = None
        
        def range_min_query(self, L, R):
            res = H+1
            stack = [(1,1,self.N)]
            while stack:
                node, l, r = stack.pop()
                self.push_down(node, l, r)
                if r < L or R < l:
                    continue
                if L <= l and r <= R:
                    res = min(res, self.minv[node])
                    continue
                mid = (l + r) >>1
                stack.append( (node<<1|1, mid+1, r) )
                stack.append( (node<<1, l, mid) )
            return res
        
        def range_assign_set(self, L, R, val):
            stack = [(1,1,self.N)]
            nodes = []
            while stack:
                node, l, r = stack.pop()
                self.push_down(node, l, r)
                if r < L or R < l:
                    continue
                if L <= l and r <= R:
                    self.lazy[node] = val
                    self.push_down(node, l, r)
                    continue
                mid = (l + r) >>1
                stack.append( (node<<1|1, mid+1, r) )
                stack.append( (node<<1, l, mid) )
            # After assignment, need to update minv upwards
            # Not implemented as we have already set minv correctly
            # because we overwrote the minv during assignment
            # So no need to do anything here

    # Use FastSegmentTree
    st = FastSegmentTree(W)
    
    results = [0]*(N+1)  # 1-based
    
    for bar in bars_sorted:
        neg_R_i, i, C_i, L_i = bar
        c1 = C_i
        c2 = C_i + L_i -1
        # Query min_last
        min_last = st.range_min_query(c1, c2)
        R_prime = min_last -1
        if R_prime < -1:
            R_prime = R_i  # unlikely
        if R_prime < -1:
            R_prime = R_i
        if R_prime <1:
            R_prime =1
        if R_prime < 0:
            R_prime = R_i
        R_prime = max(R_prime, int(-neg_R_i))
        # Assign
        st.range_assign_set(c1, c2, R_prime)
        results[i] = R_prime
    
    for i in range(1, N+1):
        print(results[i])