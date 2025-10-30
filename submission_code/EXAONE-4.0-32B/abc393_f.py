import sys

class SegmentTree:
    def __init__(self, n):
        self.n = n
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.data = [0] * (2 * self.size)
    
    def update(self, index, value):
        i = index - 1 + self.size
        if self.data[i] >= value:
            return
        self.data[i] = value
        i //= 2
        while i:
            self.data[i] = max(self.data[2*i], self.data[2*i+1])
            i //= 2
            
    def query(self, l, r):
        if l > r:
            return 0
        l0 = l - 1
        r0 = r - 1
        l0 += self.size
        r0 += self.size
        res = 0
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

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); q = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    queries = []
    all_vals = set()
    for i in range(n):
        all_vals.add(A[i])
    for i in range(q):
        R = int(next(it)); X = int(next(it))
        queries.append((R, X, i))
        all_vals.add(X)
    
    all_vals = sorted(all_vals)
    comp_map = {}
    for idx, val in enumerate(all_vals, start=1):
        comp_map[val] = idx
    M = len(all_vals)
    
    seg_tree = SegmentTree(M)
    queries_sorted = sorted(queries, key=lambda x: x[0])
    ans = [0] * q
    ptr = 0
    
    for i in range(n):
        a_val = A[i]
        c_a = comp_map[a_val]
        prev_max = seg_tree.query(1, c_a-1)
        new_val = prev_max + 1
        seg_tree.update(c_a, new_val)
        
        while ptr < len(queries_sorted) and queries_sorted[ptr][0] == i+1:
            R, X, orig_idx = queries_sorted[ptr]
            c_x = comp_map[X]
            res = seg_tree.query(1, c_x)
            ans[orig_idx] = res
            ptr += 1
            
    for a in ans:
        print(a)

if __name__ == '__main__':
    main()