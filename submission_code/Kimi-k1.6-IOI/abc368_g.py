class SumSegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.data = data.copy()
        self.build(0, 0, self.n-1)
    
    def build(self, node, start, end):
        if start == end:
            self.tree[node] = self.data[start]
        else:
            mid = (start + end) // 2
            self.build(2*node+1, start, mid)
            self.build(2*node+2, mid+1, end)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def update(self, pos, value):
        self._update(0, 0, self.n-1, pos, value)
    
    def _update(self, node, start, end, pos, value):
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            if pos <= mid:
                self._update(2*node+1, start, mid, pos, value)
            else:
                self._update(2*node+2, mid+1, end, pos, value)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def query(self, l, r):
        return self._query(0, 0, self.n-1, l, r)
    
    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return self._query(2*node+1, start, mid, l, r) + self._query(2*node+2, mid+1, end, l, r)

class ProductSegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [1] * (4 * self.n)
        self.data = data.copy()
        self.build(0, 0, self.n-1)
    
    def build(self, node, start, end):
        if start == end:
            self.tree[node] = self.data[start]
        else:
            mid = (start + end) // 2
            self.build(2*node+1, start, mid)
            self.build(2*node+2, mid+1, end)
            self.tree[node] = self.tree[2*node+1] * self.tree[2*node+2]
    
    def update(self, pos, value):
        self._update(0, 0, self.n-1, pos, value)
    
    def _update(self, node, start, end, pos, value):
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            if pos <= mid:
                self._update(2*node+1, start, mid, pos, value)
            else:
                self._update(2*node+2, mid+1, end, pos, value)
            self.tree[node] = self.tree[2*node+1] * self.tree[2*node+2]
    
    def query(self, l, r):
        return self._query(0, 0, self.n-1, l, r)
    
    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 1
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return self._query(2*node+1, start, mid, l, r) * self._query(2*node+2, mid+1, end, l, r)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr +=1
    A = list(map(int, data[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, data[ptr:ptr+N]))
    ptr +=N
    Q = int(data[ptr])
    ptr +=1
    queries = data[ptr:ptr+Q*3]
    ptr +=Q*3
    
    current_A = A.copy()
    current_B = B.copy()
    sum_st = SumSegmentTree(current_A)
    prod_st = ProductSegmentTree(current_B)
    
    output = []
    for i in range(Q):
        q_type = int(queries[i*3])
        if q_type ==1:
            idx = int(queries[i*3+1])-1
            val = int(queries[i*3+2])
            current_A[idx] = val
            sum_st.update(idx, val)
        elif q_type ==2:
            idx = int(queries[i*3+1])-1
            val = int(queries[i*3+2])
            current_B[idx] = val
            prod_st.update(idx, val)
        else:
            l = int(queries[i*3+1])-1
            r = int(queries[i*3+2])-1
            v =0
            i_idx = l
            while i_idx <=r:
                sum_A = sum_st.query(i_idx, r)
                prod_B = prod_st.query(i_idx, r)
                if prod_B ==1:
                    v += sum_A
                    break
                elif prod_B >1:
                    required = sum_A / (prod_B -1)
                    if v >= required:
                        v *= prod_B
                        break
                    else:
                        B_i = current_B[i_idx]
                        A_i = current_A[i_idx]
                        if B_i ==1:
                            v += A_i
                        else:
                            threshold = A_i / (B_i -1)
                            if v < threshold:
                                v += A_i
                            else:
                                v *= B_i
                        i_idx +=1
                else:
                    # This case is impossible since B_i >=1
                    pass
            output.append(v)
    print('
'.join(map(str, output)))

if __name__ == "__main__":
    main()