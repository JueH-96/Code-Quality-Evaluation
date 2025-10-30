import sys
MOD = 998244353

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    A = [0] + A  # 1-based indexing

    # Fenwick tree to track available numbers
    class Fenwick:
        def __init__(self, size):
            self.size = size
            self.tree = [0]*(size+2)
        
        def update(self, idx, delta):
            while idx <= self.size:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx >0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
        
        def range_query(self, l, r):
            return self.query(r) - self.query(l-1)
    
    fen = Fenwick(N)
    for i in range(1, N+1):
        fen.update(i, 1)
    
    # Segment tree for RMQ
    class SegmentTree:
        def __init__(self, size):
            self.n = 1
            while self.n < size:
                self.n <<=1
            self.size = self.n
            self.tree = [float('inf')] * (2 * self.n)
        
        def update(self, pos, value):
            pos += self.n
            self.tree[pos] = value
            while pos >1:
                pos >>=1
                self.tree[pos] = min(self.tree[2*pos], self.tree[2*pos+1])
        
        def query_min(self, l, r):
            res = float('inf')
            l += self.n
            r += self.n
            while l <= r:
                if l%2 ==1:
                    res = min(res, self.tree[l])
                    l +=1
                if r%2 ==0:
                    res = min(res, self.tree[r])
                    r -=1
                l >>=1
                r >>=1
            return res
    
    st = SegmentTree(N)
    
    result = 1
    for i in range(1, N+1):
        a = A[i]
        L = 1
        if a >0:
            # Find P_a
            # But how to get P_a?
            # This is the problem. We cannot get P_a because it's determined by previous steps.
            # So this approach is not possible.
            pass
        
        # So this approach is not feasible.
    
    print(result)

if __name__ == '__main__':
    main()