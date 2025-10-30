import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    M = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.size = 1
            while self.size < self.n:
                self.size <<= 1
            self.data = [0] * (2 * self.size)
            for i in range(self.n):
                self.data[self.size + i] = data[i]
            for i in range(self.size - 1, 0, -1):
                self.data[i] = max(self.data[2 * i], self.data[2 * i + 1])
        
        def update(self, pos, value):
            pos += self.size
            self.data[pos] = value
            while pos > 1:
                pos >>= 1
                new_val = max(self.data[2 * pos], self.data[2 * pos + 1])
                if self.data[pos] == new_val:
                    break
                self.data[pos] = new_val
        
        def query_max(self, l, r):
            res = -1
            l += self.size
            r += self.size
            while l <= r:
                if l % 2 == 1:
                    res = max(res, self.data[l])
                    l += 1
                if r % 2 == 0:
                    res = max(res, self.data[r])
                    r -= 1
                l >>= 1
                r >>= 1
            return res

    st = SegmentTree(P)
    inv_count = 0

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

    ft = FenwickTree(N)
    pos = list(range(N+1))
    current = P.copy()

    for k in A:
        max_val = st.query_max(0, k-1)
        idx = -1
        for i in range(k):
            if current[i] == max_val:
                idx = i
                break
        
        while idx != k-1:
            next_idx = idx + 1
            current[idx], current[next_idx] = current[next_idx], current[idx]
            st.update(idx, current[idx])
            st.update(next_idx, current[next_idx])
            idx = next_idx
        
        inv_count = 0
        ft = FenwickTree(N)
        temp = current.copy()
        for i in reversed(range(N)):
            inv_count += ft.query(temp[i])
            ft.update(temp[i])
        print(inv_count)

if __name__ == '__main__':
    main()