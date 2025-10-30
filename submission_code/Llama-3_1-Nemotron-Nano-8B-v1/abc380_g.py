import bisect

MOD = 998244353

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)
    
    def update(self, idx, delta=1):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def count_inversions(arr):
    n = len(arr)
    max_val = max(arr)
    ft = FenwickTree(max_val)
    inv_count = 0
    for val in reversed(arr):
        inv_count += ft.query(val - 1)
        ft.update(val)
    return inv_count % MOD

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [[] for _ in range(2 * self.size)]
        for i in range(self.n):
            self.tree[self.size + i] = [data[i]]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = sorted(self.tree[2*i] + self.tree[2*i+1])
    
    def query(self, l, r, x):
        res = 0
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res += len(self.tree[l]) - bisect.bisect_right(self.tree[l], x)
                l += 1
            if r % 2 == 0:
                res += len(self.tree[r]) - bisect.bisect_right(self.tree[r], x)
                r -= 1
            l >>= 1
            r >>= 1
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    # Compute original inversion count
    inv_count = count_inversions(P)
    
    if K == 0 or N == 0:
        print(0)
        return
    
    m = N - K + 1
    sum_count = 0
    if m > 0:
        st = SegmentTree(P)
        for j in range(N):
            L = max(0, j - K + 1)
            R = j - 1
            if L > R:
                continue
            cnt = st.query(L, R, P[j])
            sum_count = (sum_count + cnt) % MOD
    
    inv_2 = pow(2, MOD-2, MOD)
    inv_4 = pow(4, MOD-2, MOD)
    term2 = (K * (K - 1) % MOD) * inv_4 % MOD
    
    if m == 0:
        term3 = 0
    else:
        inv_m = pow(m, MOD-2, MOD)
        term3 = sum_count * inv_2 % MOD
        term3 = term3 * inv_m % MOD
    
    expected = (inv_count + term2 - term3) % MOD
    print(expected)

if __name__ == '__main__':
    main()